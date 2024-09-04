import uuid

from flask import Flask, request, Response
from flask_sock_updated import Sock
from threading import Thread, Lock

from Helper import Helper
from RequestObject import RequestObject
from CustomLogger import CustomLogger

# Shared key to make sure only auth. client is able to connect
# Also set IP restrictions in place to that only authorized client
# can connect
SHARED_KEY = "8k9JKIEP780TB56"


app = Flask(__name__)
logger = CustomLogger()
#app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your own secret key

# Sock is newer version of Socket-IO
sock = Sock(app)

#fh = Helper()

# ----------- Data Structures
requestQueue = []
responseBucket={}
webSocket = None

#@app.route('/', defaults={'path': ''})
# Define a generic route, so that the route can accomodate any custom URL
# Note:/service is not included in the path so we can pass it directly without
# any parsing
@app.route('/service/<path:path>',methods=['POST','GET'])
def defaultEntry(path):
    if("IsServiceUp" in path):
        return Response("Service is up", status=200)

    # Generate new UUID for request
    requestId = str(uuid.uuid4())
    headers = Helper.get_all_headers(request)
    payload = Helper.get_payload(request)
    requestObj = RequestObject(requestId=requestId,method=request.method,
                         headers=headers,payload=payload,urlpath=path,args=request.args)
    requestQueue.append(requestObj)
    logger.logInfo("defaultEntry",f"Response requested for {requestObj.request_id}")
    # Wait to receive response
    # async-await will be good here here but asyncio is not compat. with Flask
    while(requestId not in responseBucket):
        pass
    
    logger.logInfo("get_simple",f"Response received for {requestObj.request_id}")
    
    return Response(responseBucket[requestId].output, status=responseBucket[requestId].HTTPResponseCode)


@sock.route('/client')
async def clientWebsocket(ws):
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
    responseData = ws.receive()
    if(responseData == "HELO"+SHARED_KEY):
        ws.send("HELO-OK")
        logger.logInfo("clientWebsocket","Client connected, handshake complete")
    else:
        logger.logInfo("clientWebsocket","Invalid message from client, exiting Websocket")
        return
    
    # Start forever listen loop while client is connected
    while True:         
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
        ws.send(Helper.getSerialized(reqObject))
        responseData = ws.receive(timeout=180) # Block till data received,for 180 secs.
        responseObj =  Helper.getDeserialized(responseData)
        logger.logInfo("clientWebsocket",f"Recevied response from websocket client for {responseObj.request_id}")
        
        # Apply a lock here
        responseBucket[responseObj.request_id] = responseObj
        # release lock

async def checkRequestQueue():
    while(len(requestQueue) == 0):
        # wait for request to come from HTTP request
        print("Waiting for req ...")
        pass
    return
def show_all_headers_and_data(request):
    raw = ""
    try:
        raw = "*** Headers\n" 
        for header in request.headers:
            raw += header[0] + "::" + header[1] + "\n"
        if(request.is_json):
            jsonData = request.get_json(force=True)
        else:
            jsonData = "No JSON in request body"
        raw += "*** Body\n"
        raw += str(jsonData)
    except Exception as e:
        print("Got exception, check if request body has a JSON,",e)
    finally:
        print(raw)

if __name__ == '__main__':
    # self.app.run(host='0.0.0.0', port=5000)   
    logger.logInfo("Main","Starting webserver and websocket")
    # Give SSL context  
    # context = ""
    app.run(host='0.0.0.0')#),ssl_context=context)