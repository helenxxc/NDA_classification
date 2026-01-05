1. download the nda_app.zip file and unzip it.
2. open terminal from mac
    2.1 test if python is installed successfully using command: python3 --version (if no error message then successfully installed)
    2.2 go to the folder you download the "nda_app". e.g. nda_app is on the desktop, use command: cd desktop/nda_app
    2.3 run command: pip install -r requirements.txt
    2.4 run command: python3 app.py
        it shall return the following:
            * Serving flask app 'app'
            * Debug mode: off
            WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
            * Running on all addresses (0.0.0.0)
            * Running on http://127.0.0.1:5000
            * Running on http://192.168.1.93:5000
            Press CTRL+C to quit
            127.0.0.1 - - [25/Nov/2025 22:10:29] "GET / HTTP/1.1" 200 -
            127.0.0.1 - - [25/Nov/2025 22:10:44] "POST / HTTP/1.1" 200 -
    2.5 copy the http://127.0.0.1:5000 to web and done!

Remark: If anything above goes wrong, chatgpt is a great helper to debug. If anything goes wrong about the web application,plz contact me.