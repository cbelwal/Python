import uuid

from quart import Quart, request, Response, websocket
import asyncio

from Helper import Helper
from RequestObject import RequestObject
from CustomLogger import CustomLogger

# Shared key to make sure only auth. client is able to connect
# Also set IP restrictions in place to that only authorized client
# can connect
SHARED_KEY = "8k9JKIEP780TB56"
ALLOWED_INTERNET_IP_ADDRESSES = ["192.168.1.166","192.168.12.12"]

app = Quart(__name__)
logger = CustomLogger()
#app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your own secret key

#fh = Helper()

# ----------- Data Structures
requestQueue = []
responseQueue={}
webSocket = None

#@app.route('/', defaults={'path': ''})
# Define a generic route, so that the route can accomodate any custom URL
# Note:/service is not included in the path so we can pass it directly without
# any parsing
@app.route('/service/<path:path>',methods=['POST','GET'])
async def defaultEntry(path):
    if("IsServiceUp" in path):
        return Response("Service is up", status=200)

    if(not isIpInAllowedList()):
        return Response("Access denied", status=401)

    # Generate new UUID for request
    requestId = str(uuid.uuid4())
    headers = Helper.get_all_headers(request)
    payload = Helper.get_payload(request)
    requestObj = RequestObject(requestId=requestId,method=request.method,
                         headers=headers,payload=payload,urlpath=path,args=request.args)
    requestQueue.append(requestObj)
    logger.logInfo("defaultEntry",f"Response requested for {requestObj.request_id}")
    # Wait to receive response
    requestId = requestObj.request_id
    await checkResponseQueue(requestId)
    logger.logInfo("defaultEntry",f"Response received for {requestId}")
    
    return Response(responseQueue[requestId].output, status=responseQueue[requestId].HTTPResponseCode)


@app.websocket('/ws')
async def clientWebsocket():
    #global webSocket
    #if(webSocket == None):
    #    webSocket = ws
    # CAUTION: Only one client connection is possible 
    #else: # Previous connection is present
    #    logger.logInfo("clientWebsocket","Previous connection detected, cleaning")
        # webSocket.close() # This closed existing socket due to lack of await
        # Clear both request and response queue
    #    requestQueue.clear()
    #    responseBucket.clear()
    #    webSocket = ws
    
    # Do handshake
    responseData = await websocket.receive()
    if(responseData == "HELO"+SHARED_KEY):
        await websocket.send("HELO-OK")
        logger.logInfo("clientWebsocket","Client connected, handshake complete")
    else:
        logger.logInfo("clientWebsocket","Invalid message from client, exiting Websocket")
        return
    
    # Start forever listen loop while client is connected
    while True:   
        #thread = threading.Thread(target=checkRequestQueue)
        #thread.start()
        #thread.join()  
        #responseData = await websocket.receive()    
        await checkRequestQueue()
        #while(len(requestQueue) == 0):
            # wait for request to come from HTTP request
        #    print("Waiting for req ...")
        #    pass
        # Delete the request from the queue
        # If there is a network issue etc. the request will be lost
        # this is a reasonable compromise against more complex code, as Copilot can 
        # send the request again
        reqObject = requestQueue.pop(0) # pop(0) remove on FIFO
        logger.logInfo("clientWebsocket",f"Sending request to websocket client for {reqObject.request_id}")
        await websocket.send(Helper.getSerialized(reqObject))
        responseData = await websocket.receive() 
        responseObj =  Helper.getDeserialized(responseData)
        logger.logInfo("clientWebsocket",f"Recevied response from websocket client for {responseObj.request_id}")
        
        # Apply a lock here
        responseQueue[responseObj.request_id] = responseObj
        # release lock

async def checkRequestQueue():
    logger.logInfo("checkRequestQueue",f"Waiting for incoming requests")
    while(len(requestQueue) == 0):
        await asyncio.sleep(1)
        # wait for request to come from HTTP request
        pass
    return

async def checkResponseQueue(requestId):
    logger.logInfo("checkResponseQueue",f"Waiting for response for request {requestId}")
    while(requestId not in responseQueue):
        await asyncio.sleep(.5)
        pass
    return


def isIpInAllowedList(request):
    requestIp = str(getIpAddr(request)).strip()

    if(requestIp in ALLOWED_INTERNET_IP_ADDRESSES):
        return True
    return False


def getIpAddr(request):
    if request.headers.getlist("X-Forwarded-For"):
        ip_addr = request.headers.getlist("X-Forwarded-For")[0]
    else:
        ip_addr = request.remote_addr
    return ip_addr

if __name__ == '__main__': 
    logger.logInfo("Main","Starting webserver and websocket")
    # Give SSL context  
    # context = ""
    app.run(host='0.0.0.0')#),ssl_context=context)