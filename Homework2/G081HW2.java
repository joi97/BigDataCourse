import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;
import scala.Tuple2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.lang.Math;

public class G081HW2 {


    public static void main(String[] args) throws IOException {

        //checking arguments
        if (args.length != 3)
            throw new IllegalArgumentException("USAGE: path_file K_number_of_cluster Z_farthest_point_to_remove"); //expected_sample_size_per_cluster

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //SPARK SETUP
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("HW2");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //INPUT READING
        //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        //Read input file and subdivide it in 5 random partitions

        ArrayList<Vector> inputPoints = readVectorsSeq(args[0]);
        System.out.println("Input size n = " + inputPoints.size());
        ArrayList<Long> weights = new ArrayList<Long>(Collections.nCopies(inputPoints.size(), 1L));

        //Read number of cluster
        int K = Integer.parseInt(args[1]);
        System.out.println("Number of centers k = " + K);

        //Read number of removing samples
        int Z = Integer.parseInt(args[2]);
        System.out.println("Number of outliers z = " + Z);

        long startTime = System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, K, Z, 0);
        long stopTime = System.currentTimeMillis();

        System.out.println("time: " + (stopTime - startTime));

        System.out.println("solution: "+solution);
        double objective = ComputeObjective(inputPoints,solution,Z);


    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Input reading methods
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename)).map(str -> strToVector(str)).forEach(e -> result.add(e));
        return result;
    }


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Input reading methods
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



    public static double ComputeObjective(ArrayList<Vector> P,ArrayList<Vector> S,int z){
        double result = 0;
        double euclidDistance;

        Map<Vector,  List<Double>> centerDistance = new HashMap<>();


        for (Vector vectorX : S) {
            List<Double> distancesFromCenter = new ArrayList<>();
            for (Vector vectorY : P) {
                if (vectorX == vectorY) continue;
                euclidDistance = Math.sqrt(Vectors.sqdist(vectorX, vectorY));
                distancesFromCenter.add(euclidDistance);



            }

            Collections.sort(distancesFromCenter);
            centerDistance.put(vectorX, distancesFromCenter);


        }

        System.out.println("centerDistance");
        System.out.println(centerDistance);

        return result;
    }


    public static ArrayList<Vector> SeqWeightedOutliers(ArrayList<Vector> inputPoints, ArrayList<Long> weights, int k, int z, int alfa) {

        ArrayList<Double> all_distances = new ArrayList<>();
        double euclidDistance;
        int indexR = 1;

        //calculate first Rmin
        Vector initialVectorX = inputPoints.get(0);
        for (Vector vectorY : inputPoints) {

            //same point so I skip it
            if (initialVectorX == vectorY) continue;

            //calculate Euclidean Distance
            euclidDistance = Math.sqrt(Vectors.sqdist(initialVectorX, vectorY));

            if (euclidDistance <= (k + z + 1)) {
                all_distances.add(euclidDistance);
            }

        }

        //take the min r
        Collections.sort(all_distances);
        double r = all_distances.get(0) / 2;
        System.out.println("initial guess= " + r);

        while (true) {

            //Z <- P
            ArrayList<Vector> Z = new ArrayList<>(inputPoints);

            //S = 0
            ArrayList<Vector> result = new ArrayList<>();

            //inizialize Weights
            Long Wz = 0L;
            for (Long d : weights)
                Wz += d;

            while ( (result.size() < k) && (Wz > 0) ) {

                double max = 0L;
                Vector newcenter = null;

                // foreach x ∈ P do
                for (Vector vectorX : inputPoints) {//Z

                    //calculate ball_weight
                    double ball_weight = 0;


                    for (Vector vectorY : Z) { //Z

                        euclidDistance = Math.sqrt(Vectors.sqdist(vectorX, vectorY));

                        if (euclidDistance <= ((1 + 2 * alfa) * r)) {
                            ball_weight = ball_weight + weights.get(Z.indexOf(vectorY));
                        }

                    }

                    //I'm going to find my max that maximize the max values of balls that I can cover

                    if (ball_weight > max) {
                        max = ball_weight;
                        newcenter = vectorX;
                    }

                }

                result.add(newcenter);  //S = S ∪ {new center};

                ArrayList<Vector> pointsToRemove = new ArrayList<>();

                //foreach (y ∈ BZ (new center,(3 + 4α)r)) do
                for (Vector vectorY : Z) {//Z
                    euclidDistance = Math.sqrt(Vectors.sqdist(newcenter, vectorY));
                    if (euclidDistance <= ((3 + 4 * alfa) * r)) {
                        Wz = Wz - weights.get(Z.indexOf(vectorY));
                        pointsToRemove.add(vectorY);
                    }
                }
                for (Vector v : pointsToRemove) {
                    Z.remove(v);
                }

            }


            if (Wz <= z) {
                System.out.println("final guess= " + r);
                System.out.println("Number of guesses= " + indexR);
                return result;
            } else {
                r = 2 * r;
                indexR = indexR+1;
            }


        }

    }

}
