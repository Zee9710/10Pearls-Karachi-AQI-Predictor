import os
from flask import Flask, send_from_directory
from flask_cors import CORS

_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist"
)


def create_app():
    app = Flask(__name__)
    CORS(app)
    from api.routes import bp
    app.register_blueprint(bp)

    # Single-origin: serve the built frontend. /api/* is matched by the
    # blueprint (more specific rules win), everything else falls through to the
    # SPA so client-side routing and deep links resolve to index.html.
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        target = os.path.join(_DIST, path)
        if path and os.path.isfile(target):
            return send_from_directory(_DIST, path)
        return send_from_directory(_DIST, "index.html")

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
