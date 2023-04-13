from flask import Flask, render_template
from tremor.hand_tremor import data
import subprocess
app = Flask(__name__)

# Define route for home page
@app.route("/")
def home():
    return render_template('index.html')
#route for instruction Page
@app.route("/instruction")
def instructionPage():
    return render_template('instruction.html')
#route for tremor test
@app.route("/Tip")
def tipPage():
    testType = {'type':'TIP','nextPage':'Dip'}
    return render_template('tremorTest.html',testType=testType)
@app.route("/Dip")
def dipPage():
    testType = {'type':'DIP','nextPage':'Pip'}
    return render_template('tremorTest.html',testType=testType)
@app.route("/Pip")
def pipPage():
    testType = {'type':'PIP','nextPage':'Mcp'}
    return render_template('tremorTest.html',testType=testType)
@app.route("/Mcp")
def mcpPage():
    testType = {'type':'MCP','nextPage':'spiral_wave'}
    return render_template('tremorTest.html',testType=testType)

@app.route("/tremor")
def run_tremor():
    try:
        subprocess.run(["python",'tremor/hand_tremor.py'])
        if data:
            return data
    except subprocess.CalledProcessError:
        return "ERROR"
    
#spiral and wave test route
@app.route("/spiral_wave")
def run_spiral():
    try:
        subprocess.run(["python","spiralTest/app.py"])
        return "Success"
    except subprocess.CalledProcessError:
        return "error"
if __name__ == "__main__":
    app.run(debug=True)
