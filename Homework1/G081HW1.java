import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G081HW1 {

    public static void main(String[] args) throws IOException {

        SparkConf conf = new SparkConf(true).setAppName("HomeWork1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");


        int K = Integer.parseInt(args[0]);
        int H = Integer.parseInt(args[1]);
        String S = new String(args[2]);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 1° EXERCISE
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();

        long numdocs, numProductCustomer;
        numdocs = rawData.count();
        Random randomGenerator = new Random();
        System.out.println("Number of rows = " + numdocs);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 2° EXERCISE
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        JavaPairRDD<String, Integer> productCustomer = rawData
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = document.split(",");
                    List<Tuple2<Tuple2<String, Integer>, Integer>> pairs = new ArrayList<>();
                    if (Integer.parseInt(tokens[3]) > 0) {
                        if (S.equals("all") || S.equals(tokens[7])) {
                            pairs.add(new Tuple2<>(new Tuple2<>(tokens[1], Integer.parseInt(tokens[6])), 0));
                        }
                    }

                    return pairs.iterator();
                })
                .groupByKey()  // <-- REDUCE PHASE (R1)
                .flatMapToPair((prodCustomer) -> { //MAP PHASE(R2)

                    List<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    pairs.add(new Tuple2<>(prodCustomer._1._1, prodCustomer._1._2));

                    return pairs.iterator();
                });

        numProductCustomer = productCustomer.count();
        System.out.println("Product-Customer Pairs = "+numProductCustomer);


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 3° EXERCISE
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // I want an RDD String,Integer that for each product gives me the number of different Customers
        JavaPairRDD<String, Integer> productPopularity1 = productCustomer
                .mapPartitionsToPair((productBoughtSameCustomer) -> {
                    HashMap<String, Integer> productDiffCustomer = new HashMap<>();

                    while (productBoughtSameCustomer.hasNext()) {
                        Tuple2<String, Integer> tuple = productBoughtSameCustomer.next();
                        productDiffCustomer.put(tuple._1, 1 + productDiffCustomer.getOrDefault(tuple._1, 0));
                    }

                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Integer> e : productDiffCustomer.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- SHUFFLE+GROUPING
                .mapValues((it) -> { // <-- REDUCE PHASE (R2)
                    int sum = 0;
                    for (int c : it) {
                        sum += c;
                    }
                    return sum;
                });

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 4° EXERCISE
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Integer> productPopularity2 = productCustomer
                .reduceByKey(Integer::sum);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 6° EXERCISE PRINT productPopularity1, productPopularity2
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (H == 0) {
            System.out.println("productPopularity1:");
            productPopularity1 = productPopularity1.sortByKey();
            List<Tuple2<String, Integer>> productPopularity1List = new ArrayList<>();

            for (Tuple2<String, Integer> line : productPopularity1.collect()) {
                productPopularity1List.add(line);
            }

            for (Tuple2<String, Integer> elem : productPopularity1List)
                System.out.print("Product: " + elem._1 + " Popularity: " + elem._2 + "; ");

            System.out.println("\nproductPopularity2:");

            productPopularity2 = productPopularity2.sortByKey();
            List<Tuple2<String, Integer>> productPopularity2List = new ArrayList<>();

            for (Tuple2<String, Integer> line : productPopularity2.collect()) {
                productPopularity2List.add(line);
            }

            for (Tuple2<String, Integer> elem : productPopularity2List)
                System.out.print("Product: " + elem._1 + " Popularity: " + elem._2 + "; ");
        } else {


            // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            // 5° EXERCISE PRINT TOP H PRODUCT
            // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            System.out.println("Top " + H + " Products and their Popularities");
            List<Tuple2<String, Integer>> topProduct = productPopularity1.mapToPair(x -> x.swap()).sortByKey(false).mapToPair(x -> x.swap()).take(H);
            for (int i = 0; i < H; i++)
                System.out.print("Product " + topProduct.get(i)._1 + " Popularity " +  topProduct.get(i)._2 + "; ");

        }

    }

}
