import json

class RequestObject:
    def __init__(self,requestId,method,headers,payload,urlpath,args):
        self.method = method
        self.request_id = requestId
        self.headers = headers
        self.payload = payload
        self.urlpath = urlpath 
        self.args = args

    def getSerializedJSON(self):
        return json.dumps(self,default=lambda o: o.__dict__)
