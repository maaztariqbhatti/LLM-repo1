import uvicorn

if __name__ == "__main__":
    uvicorn.run("api_endpoint:app", host="192.168.9.155", port=8093, reload="true")
