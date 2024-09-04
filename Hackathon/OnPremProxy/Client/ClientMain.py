import asyncio
import jsonpickle
from Helper import Helper
from websockets.asyncio.client import connect

from RequestObject import RequestObject
from ResponseObject import ResponseObject
from CustomLogger import CustomLogger
from APICaller import APICaller

# This key should match the one in the server
SHARED_KEY = "redacted"

URL_INTERNAL_API_SERVER = "http://192.168.1.201:5000/"
URL_PROXY_API_SERVER  = "ws://localhost:5000/ws"

logger = CustomLogger()


async def startClient():
    apiCaller = APICaller(URL_INTERNAL_API_SERVER)
    uri = URL_PROXY_API_SERVER
    async with connect(uri) as websocket:
        logger.logInfo("startClient","Client started")
        
        # Do handshake
        await websocket.send("HELO" + SHARED_KEY)
        responseData = await websocket.recv()
        
        if(responseData == "HELO-OK"):
            logger.logInfo("startClient","Server connected, handshake completed")
        else:
            logger.logInfo("Invalid message from client, exiting Websocket")
            exit()
        
        # Start listen loop
        try:
            while True:
                requestData = await websocket.recv()
                requestObj = Helper.getDeserialized(requestData)
                logger.logInfo("startClient",f"Received request::{requestObj.request_id}")
                responseObj = apiCaller.makeAPICall(requestObj=requestObj)
                
                await websocket.send(Helper.getSerialized(responseObj))
        except Exception as e:
                logger.logError("startClient",f"Exception in Client {e}")
        finally:
             await websocket.close()

if __name__ == "__main__":
    asyncio.run(startClient())