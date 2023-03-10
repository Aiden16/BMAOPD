from flask import Flask, render_template

app = Flask(__name__)

# Define route for home page
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/instruction")
def instructionPage():
    return render_template('instruction.html')

@app.route("/test1")
def test1Page():
    return render_template('tremorTest.html')
if __name__ == "__main__":
    app.run(debug=True)
