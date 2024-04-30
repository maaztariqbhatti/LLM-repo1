import uvicorn

if __name__ == "__main__":
    uvicorn.run("llama13bChat_MLServer:app", host="192.168.9.155", port=8093, reload="true")
