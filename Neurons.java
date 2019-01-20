/*
 * Neurons.java
 * Нейронная сеть
 * © N-Creative, 2019. Доступен по лицензии MIT
 */

package detrain;

//TODO: Debug it!
public class Neurons {    
    protected double[] input;
    protected double[] hidden;
    protected double output;
    
    protected double[] offsetSA; //смещения сигналов от входа к скрытому слою
    protected double[][] weightSA; //веса от входа к скрытому слою
    protected double offsetAR; //смещение сигнала от скрытого слоя к выходу
    protected double[] weightAR; //веса от скрытого слоя к выходу
    
    Neurons (int in, int hdn) {
        input = new double[in];
        hidden = new double[hdn];
        weightAR = new double[hdn];
        offsetSA = new double[hdn];
        weightSA = new double[in][hdn];
        
        //матрица весов: строки - нейроны текущего слоя, столбцы - нейроны следующего слоя
        offsetAR = Math.random() - 0.5; //инициализация смещения AR
        for (int h = 0; h < hdn; h++) {
            for (int i = 0; i < in; i++)
                weightSA[i][h] = Math.random() - 0.5; //инициализация весов SA
            
            offsetSA[h] = Math.random() - 0.5; //инициализация смещений SA
            weightAR[h] = Math.random() - 0.5; //инициализация весов AR
        }
    }
    
    //Запуск нейронной сети
    public double run(double[] data, boolean train) {
        if (input.length == data.length) {           
            //распространение сигналов по SA-дендритам
            for (int h = 0; h < hidden.length; h++) {
                hidden[h] = offsetSA[h];
                for (int i = 0; i < data.length; i++)
                    hidden[h] += data[i] * weightSA[i][h];
            }
            
            //распространение сигналов по AR-дендритам
            output = sigmoid(offsetAR);
            for (int h = 0; h < hidden.length; h++)
                output += sigmoid(hidden[h]) * weightAR[h];
            
            if (train) input = data;
            
            return output;
        }
        
        return 0;
    }
    
    //обучение методом обратного распространения ошибки
    public void train(double t, double output, double a) {
        double doAR;
        double[] dwAR = new double[hidden.length];
        double[] doSA = new double[hidden.length];
        double[][] dwSA = new double[input.length][hidden.length];
        
        //распространение от выхода к скрытым слоям
        double dout = t - output;
        doAR = a * dout;
        for (int h = 0; h < hidden.length; h++)
            dwAR[h] = doAR * sigmoid(hidden[h]);
        
        //распространение от скрытых слоёв ко входу
        for (int h = 0; h < hidden.length; h++) {
            doSA[h] = a * dout * weightAR[h] * dSigmoid(hidden[h]);
            for (int i = 0; i < input.length; i++) {
                dwSA[i][h] = doSA[h] * input[i];
            }
        }
        
        //правим веса и смещения
        for (int h = 0; h < hidden.length; h++)
            for (int i = 0; i < input.length; i++)
                weightSA[i][h] += dwSA[i][h];
        
        for (int h = 0; h < hidden.length; h++)
            offsetSA[h] += doSA[h];
        
        for (int h = 0; h < hidden.length; h++) 
            weightAR[h] += dwAR[h];
        
        offsetAR += doAR;
    }
    
    protected double sigmoid(double x) {
        return (2 / (1 + Math.exp(-x)) - 1);
    }
    
    protected double dSigmoid(double x) {
        return ((1 - Math.pow(sigmoid(x), 2)) / 2);
    }
}