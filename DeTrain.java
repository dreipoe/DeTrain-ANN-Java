package detrain;

public class DeTrain {

    public static void main(String[] args) {        
        Neurons ann = new Neurons(2, 3);
        
        short[][] train = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        System.out.println("Training of test ANN...\n");
        
        double t, output;
        for (int i = 0; i < 100; i++) {
            for (short[] row : train) {
                t = row[0] + row[1];
                double[] drow = {row[0], row[1]};
                output = ann.run(drow, true);
                ann.train(t, output, 1);
            }
        }
        
        System.out.println("Testing ANN...\n");
        
        for (short[] row : train) {
            double[] drow = {row[0], row[1]};
            output = Math.round(ann.run(drow, false));
            System.out.printf("%d + %d = %.3f\n", row[0], row[1], output);
        }
    }   
}
