import jsonpickle

class Helper:
    def get_all_headers(request):
        headers={}
        try:            
            for header in request.headers:
                headers[header[0]] = header[1]
        except Exception as e:
            print("Got exception, check if request body has a JSON,",e)
        finally:
            return headers
        
    def get_payload(request):
        jsonData = ""
        try:
            if(request.is_json):
                jsonData = request.get_json(force=True)
        except Exception as e:
            print("Got exception, check if request body has a JSON,",e)
        finally:
            return jsonData
    
    def getSerialized(obj):
        return jsonpickle.encode(obj)
    
    def getDeserialized(serializedStr):
        return jsonpickle.decode(serializedStr)
    
    