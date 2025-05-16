from flask import Flask, render_template, request
from predict import predict_affinity

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        drug = request.form['drug']
        protein = request.form['protein']
        output = predict_affinity(drug, protein)

        if output is None:
            error = "❌ Prediction failed. Please check the drug or protein name."
        else:
            chembl_id, uniprot_id, score = output
            result = {
                'drug': drug,
                'chembl_id': chembl_id,
                'protein': protein,
                'uniprot_id': uniprot_id,
                'score': round(score, 4)
            }

    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
