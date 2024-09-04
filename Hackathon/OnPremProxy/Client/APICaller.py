import requests
from RequestObject import RequestObject
from ResponseObject import ResponseObject

'''
This class will handle making API requests to the on-prem instance
and pass the response back. It will frame the API requests based on the data 
in the RequestObject
'''
class APICaller:
    def __init__(self, instanceURL):
        self.instanceURL = instanceURL

    '''
    Makes an API Call
    Take an input the requestObj and 
    sends response wht responseObj 
    '''
    def makeAPICall(self, requestObj:RequestObject):
        url = (str)(self.instanceURL + requestObj.urlpath)
        
        
        if(requestObj.method == "GET"):
            response = requests.get(url, data=requestObj.payload,
                                    headers=requestObj.headers, params=requestObj.args, verify=True)
        
        responseObj = ResponseObject(requestId=requestObj.request_id,
                                     HTTPCode=response.status_code,output=response.text)
        #elif (requestObj.method == "POST"):
        #    response = requests.post(url, data=requestObj.payload,headers=,verify=True)
        return responseObj