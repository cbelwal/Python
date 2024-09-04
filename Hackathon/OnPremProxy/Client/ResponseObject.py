class ResponseObject:
    def __init__(self,requestId,output, HTTPCode):
        self.request_id = requestId
        self.output = output
        self.HTTPResponseCode = HTTPCode