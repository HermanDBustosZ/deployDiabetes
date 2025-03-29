from fastapi import FastAPI

from routers import routerPredictions


app = FastAPI()
app.include_router(routerPredictions.router)


if __name__ == "__main__":
    app.run()