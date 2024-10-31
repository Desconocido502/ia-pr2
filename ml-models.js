document.getElementById("file-input").addEventListener("change", handleFileUpload);
document.getElementById("model-select").addEventListener("change", resetModel);

let xTrain = [];
let yTrain = [];
let xToPredict = [];
let model;

// Función para limpiar datos al cambiar de modelo
function resetModel() {
    xTrain = [];
    yTrain = [];
    xToPredict = [];
    model = null;
    document.getElementById("logRS").innerHTML = "";
    if (window.myChart) {
        window.myChart.destroy();
    }

    // Muestra el selector de grado si es una regresión polinomial
    const modelType = document.getElementById("model-select").value;
    document.getElementById("degree-select-group").style.display = modelType === "polynomial-regression" ? "block" : "none";
}


function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            parseCSVData(csvData);
        };
        reader.readAsText(file);
    }
}

// Parseo del CSV
function parseCSVData(data) {
    const modelType = document.getElementById("model-select").value;
    console.log("modelType: ", modelType);
    const lines = data.trim().split('\n').slice(1);
    xTrain = [];
    yTrain = [];
    xToPredict = [];

    if (modelType == "linear-regression") {
        lines.forEach(line => {
            const [x, y] = line.split(';').map(parseFloat);
            xTrain.push(x);
            yTrain.push(y);
        });
    } else if (modelType == "polynomial-regression") {
        lines.forEach(line => {
            const [x, y, xPred] = line.split(';').map(parseFloat);
            xTrain.push(x);
            yTrain.push(y);
            xToPredict.push(xPred);
        });
        console.log("xToPredict:", xToPredict);

    }
    console.log("XTrain:", xTrain);
    console.log("YTrain:", yTrain);


}

// Clase de Regresión Lineal
class LinearRegression {
    constructor() {
        this.m = 0;
        this.b = 0;
        this.isFit = false;
    }

    fit(xTrain, yTrain) {
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

        if (xTrain.length !== yTrain.length) {
            throw new Error('Los parámetros para entrenar no tienen la misma longitud!');
        }

        for (let i = 0; i < xTrain.length; i++) {
            sumX += xTrain[i];
            sumY += yTrain[i];
            sumXY += xTrain[i] * yTrain[i];
            sumXX += xTrain[i] * xTrain[i];
        }

        const n = xTrain.length;
        this.m = (n * sumXY - sumX * sumY) / (n * sumXX - Math.pow(sumX, 2));
        this.b = (sumY * sumXX - sumX * sumXY) / (n * sumXX - Math.pow(sumX, 2));
        this.isFit = true;
    }

    predict(xTest) {
        return this.isFit ? xTest.map(x => this.m * x + this.b) : [];
    }
}

// Clase de Regresión Polinomial
class PolynomialRegression {
    //PolynomialRegression code
    constructor() {
        this.coefficients = [];
        this.isFit = false;
    }

    //Method that trains the model in order to create the regression
    fit(xArray, yArray, degree) {
        //Equation matrix size based on the degree and number of elements
        let equationSize = degree + 1;
        let nElements = degree + 2;

        //Equation matrix to be solved
        let equations = new Array(equationSize);
        for (let i = 0; i < equationSize; i++) {
            equations[i] = new Array(nElements);
        }

        //Building equation matrix
        for (let i = 0; i < equationSize; i++) {
            for (let j = 0; j < nElements; j++) {
                let sum = 0;
                if (i == 0 && j == 0) {
                    sum = xArray.length;
                }
                else if (j == nElements - 1) {
                    for (let k = 0; k < xArray.length; k++) {
                        sum += Math.pow(xArray[k], i) * yArray[k];
                    }
                }
                else {
                    for (let k = 0; k < xArray.length; k++) {
                        sum += Math.pow(xArray[k], (j + i));
                    }
                }
                equations[i][j] = sum;
            }
        }

        //Staggering matrix
        for (let i = 1; i < equationSize; i++) {
            for (let j = 0; j <= i - 1; j++) {
                let factor = equations[i][j] / equations[j][j];
                for (let k = j; k < nElements; k++) {
                    equations[i][k] = equations[i][k] - factor * equations[j][k];
                }
            }
        }

        //Solving matrix
        for (let i = equationSize - 1; i > -1; i--) {
            for (let j = equationSize - 1; j > -1; j--) {
                if (i == j) {
                    equations[i][nElements - 1] = equations[i][nElements - 1] / equations[i][j];
                }
                else if (equations[i][j] != 0) {
                    equations[i][nElements - 1] -= equations[i][j] * equations[j][nElements - 1];
                }
            }
        }

        //Storing solutions
        this.solutions = new Array(equationSize);
        for (let i = 0; i < equationSize; i++) {
            this.solutions[i] = equations[i][nElements - 1];
        }

        //Setting Model as trained
        this.isFit = true;

        //Setting error
        this.calculateR2(xArray, yArray);
    }

    //Function that creates a prediction based in the regression model
    predict(xArray) {
        let yArray = [];
        //Checking if the model is already trained
        if (this.isFit) {
            //Generating the predictions based in the input and solutions
            for (let i = 0; i < xArray.length; i++) {
                let yprediction = 0;
                for (let j = 0; j < this.solutions.length; j++) {
                    yprediction += this.solutions[j] * Math.pow(xArray[i], j);
                }
                yArray.push(yprediction);
            }
        }

        //Returning Prediction
        return yArray;
    }

    //Method that stores error for the trained array
    calculateR2(xArray, yArray) {
        //Setting error array and predictions
        let errors = new Array(xArray.length);
        let prediction = this.predict(xArray);
        let sumY = 0;

        //Calculating errors
        for (let i = 0; i < xArray.length; i++) {
            sumY += yArray[i];
            errors[i] = Math.pow(yArray[i] - prediction[i], 2);
        }

        let sr = 0;
        let st = 0;
        for (let i = 0; i < xArray.length; i++) {
            sr += errors[i];
            st += Math.pow(yArray[i] - (sumY / xArray.length), 2);
        }
        let r2 = (st - sr) / st;
        this.error = r2;
    }

    getError() {
        return this.error;
    }
}

// Función para dividir datos en entrenamiento y prueba
function splitData(trainPercent) {
    const trainSize = Math.floor(xTrain.length * (trainPercent / 100));
    const xTrainSplit = xTrain.slice(0, trainSize);
    const yTrainSplit = yTrain.slice(0, trainSize);
    const xTestSplit = xTrain.slice(trainSize);
    const yTestSplit = yTrain.slice(trainSize);

    return { xTrainSplit, yTrainSplit, xTestSplit, yTestSplit };
}

// Entrenar el modelo
function trainModel() {
    const modelType = document.getElementById("model-select").value;
    const trainPercent = parseInt(document.getElementById("train-test-split").value);
    console.log("trainPercent: ", trainPercent);
    const { xTrainSplit, yTrainSplit } = splitData(trainPercent);

    if (modelType === "linear-regression") {
        model = new LinearRegression();
        model.fit(xTrainSplit, yTrainSplit);

        let yPredict = model.predict(xTrain);
        yPredict = yPredict.map(value => Number(value.toFixed(2)));
        console.log(yPredict);
        document.getElementById("logRS").innerHTML = `
            <strong>Datos de Entrenamiento:</strong><br>
            X Train: ${xTrainSplit}<br>
            Y Train: ${yTrainSplit}<br>
            Y Predict: ${yPredict}
            `;
        drawChart(xTrain, yTrain, yPredict);
    } else if (modelType === "polynomial-regression") {
        const degree = parseInt(document.getElementById("degree-select").value);
        model = new PolynomialRegression();
        model.fit(xTrainSplit, yTrainSplit, degree);

        console.log("xToPredict tain:", xToPredict);

        let yPredict = model.predict(xToPredict);
        yPredict = yPredict.map(value => Number(value.toFixed(2)));
        //r2 = model.getError();
        //console.log("r2:", r2);
        displayResults(xToPredict, yTrain, yPredict);
    } else {
        console.warn("Seleccione un modelo válido para entrenar.");
    }
}

// Mostrar resultados y graficar
function displayResults(x, yTrain, yPredict) {
    document.getElementById("logRS").innerHTML = `
      <strong>Datos de Entrenamiento:</strong><br>
      X Train: ${x}<br>
      Y Train: ${yTrain}<br>
      Y Predict: ${yPredict}
    `;
    drawChart(x, yTrain, yPredict);
}

// Función para realizar predicción en un nuevo rango
function makePrediction() {
    if (!model || !model.isFit) {
        alert("Primero entrena el modelo.");
        return;
    }

    const rangeInput = document.getElementById("prediction-range").value;
    const newRange = rangeInput.split(',').map(parseFloat);
    const yPredictions = model.predict(newRange);

    document.getElementById("logRS").innerHTML += `<br><strong>Nueva Predicción:</strong> X: ${newRange}, Y: ${yPredictions}`;
}

// Gráfico usando Chart.js
function drawChart(xTrain, yTrain, yPredict) {
    const ctx = document.getElementById('chart_divRS').getContext('2d');

    if (window.myChart) {
        window.myChart.destroy();
    }

    const data = {
        labels: xTrain,
        datasets: [
            {
                label: 'Datos de Entrenamiento',
                data: yTrain,
                borderColor: 'blue',
                backgroundColor: 'blue',
                type: 'scatter',
                pointRadius: 4,
            },
            {
                label: 'Predicción',
                data: yPredict,
                borderColor: 'red',
                fill: false,
                type: 'line',
            }
        ]
    };

    const options = {
        responsive: true,
        scales: {
            x: { title: { display: true, text: 'X' } },
            y: { title: { display: true, text: 'Y' } }
        }
    };

    window.myChart = new Chart(ctx, {
        type: 'scatter',
        data: data,
        options: options
    });
}
