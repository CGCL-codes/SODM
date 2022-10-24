import ODMModel.ODMModel
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import java.io.FileWriter

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object ODMTest {
  def main(args: Array[String]): Unit ={
    val conf = new SparkConf().setAppName("ODMModel")
      .setMaster("local")
      .set("spark.speculation", "true")
      .set("spark.speculation.interval", "300s")
      .set("spark.speculation.quantile","0.9")

    val sc = new SparkContext(conf)
    //val textFile1 = sc.textFile("DCODM810_3.txt")
    val textFile = sc.textFile("susy_train.txt")
    //val textFile = sc.textFile("hdfs://node18:9000/user/wangyilin/a7a-1.txt")
    //val textFile = sc.textFile("hdfs://master:51234/user/docker/DLA/data.txt")
    val data  = textFile.map{ line =>
      val num = line.split(',').map(_.toDouble)
      val label = num.apply(num.length-1).toInt
      num(num.size-1) = 1
      val vec = DenseVector(num)//最后一个是标签，1或-1
      (vec, label)
      //(vec, vec.apply(vec.size - 1).toInt)
    }.cache()
    //.sample(false, 0.02).cache()//读入数据，以空格隔开转为Double
    val m = data.count()//数据条数
    val sample = data.take(1)
    val n = sample(0)._1.length//特征维度
    val model = new ODMModel(0.5, 32.0, 0.5, m, n, 1e-6, 0.0001)
    //val testData = data.sample(false, 0.5)
    //val w = model.svrgTrain(data, 20, 10, 1, 20, 0.2)
    //val w = model.dsvrgTrain(data, 20, 10)
    //val w = model.dsvrgWithKmeans(data, 30, 100)
    //print(w)
    /*
    println(try{
      val out = new FileWriter("test.txt",true)
      for(x <- w) {
        out.write(x.toString+',')
      }
      out.write('\n')
      out.close()
    })
    */
    //val V = model.DCSolver(data, 30, 4, 0.1)
    //val V = model.RandomSolver(data, 30, 4)

    val V = model.KmeansSolver(data, 30, 6, 0.5, 200)
    //val V = model.DPPSolver(data, 30, 6, 0.5, 200)
    //val V = model.CascadeSolver(data, 30, 6)

    //val localData = data.map(_._1).collect()
    //val X = DenseMatrix.tabulate[Double](n.toInt, m.toInt) {case(i, j) => localData(j).apply(i)}
    //val w = X * V

    /*
    val v = textFile1.map{ line =>
      val num = line.split(',').map(_.toDouble)

      val vec = DenseVector(num)//最后一个是标签，1或-1
      vec
      //(vec, vec.apply(vec.size - 1).toInt)
    }.collect()
    val V = v(0)
    */



    //val w = V


    //val V = DenseVector[Double](data.map(_._2.toDouble).collect())
    /*
    val x = data.map(_._1).collect()
    val X = DenseMatrix.zeros[Double](x.head.length, x.length)
    var i = 0
    x.foreach{ v =>
      X(::,i) := v
      i += 1
    }
    */


    /*
    println(try{
      val out = new FileWriter("test03.txt",true)
      for(x <- w) {
        out.write(x.toString+',')
      }
      out.write('\n')
      out.close()
    })
    */



  }
}
