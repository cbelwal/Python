**Server Proxy**

app.py is the main code that runs an asyc webservice and website using quart library.
quart is built on top of Flask.

Make sure the SHARED_KEY specified here is same as specified in client.

On receiving the request from Copilot, this server takes the URL path and headers and without any modification puts them in RequestObject class, and sends the serialized object to the client.




