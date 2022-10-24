package ODMModel

import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable._
import breeze.numerics._
import org.apache.spark.mllib.linalg.BLAS
import java.io.FileWriter
import org.apache.spark.TaskContext
import scala.util.Random
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs
import org.apache.hadoop.fs.Path
import java.io.BufferedOutputStream


class ODMModel (
                 private var thet: Double,//ODM参数，为减少支持向量的个数设置的偏差容忍值
                 private var lambda: Double,//ODM参数
                 private var mu: Double,//ODM参数
                 private var m: Long,//训练数据数量
                 private var n:Int,//训练数据维度
                 private var exps: Double,//参数更新终止阈值
                 //private var maxIter: Int,
                 //private var frequency: Int,
                 private var eta: Double//DSVRG步长
               ) extends  Serializable with Logging {

  /**多节点计算函数梯度，主要用于SVRG中计算全梯度*/
  def gradient(w3: DenseVector[Double], x: RDD[(DenseVector[Double], Int)]): DenseVector[Double] = {
    val sc = x.sparkContext
    val bv = sc.broadcast((w3, thet,lambda,m,mu,n))//广播相关配置
    val gradient = x.mapPartitions{ iter =>
      val w3 = bv.value._1
      val thet = bv.value._2
      val lambda = bv.value._3
      val m = bv.value._4
      val miu = bv.value._5
      val n = bv.value._6
      var g = DenseVector.zeros[Double](n)
      iter.foreach{ v =>
        val feature = v._1 //v._1为xi，xi与数据特征同维度
        val label = v._2.toDouble  //为yi
        if ((w3 dot feature) * label < 1 - thet) {  //对应文献中I2 = {i|yi*w*xi<1-thet}
          axpy((2*lambda*(label*(feature dot w3)+thet-1)/(m*(1-thet)*(1-thet)))*label.toDouble,feature,g)
        }
        else if ((w3 dot feature) * label > 1 + thet) {  //对应文献中I3 = {i|yi*w*xi > 1 + thet}
          axpy((2 * lambda * miu * (label * (feature dot w3) - thet - 1) / (m * (1 - thet) * (1 - thet)) )* label.toDouble, feature, g)
        }
      }
      Array(g).iterator
    }.reduce{ (a,b) =>
      axpy(1.0, a, b)//b = a*x+b b=a+b 这里b为gradient
      b
    }
    axpy(1.0, gradient, w3)//w = gradient + w
    w3
  }

  def lossB(w3: DenseVector[Double], x: RDD[(DenseVector[Double], Int)]): DenseVector[Double] = {
    val sc = x.sparkContext
    val bv = sc.broadcast((w3, thet,lambda,m,mu,n))//广播相关配置
    val gradient = x.mapPartitions{ iter =>
      val w3 = bv.value._1
      val thet = bv.value._2
      val lambda = bv.value._3
      val m = bv.value._4
      val miu = bv.value._5
      val n = bv.value._6
      var g = DenseVector.zeros[Double](n)
      //      val loB = 0.0
      iter.foreach{ v =>
        //        val feature = v._1 //v._1为xi，xi与数据特征同维度
        val feature = DenseVector.ones[Double](n)
        val label = v._2.toDouble  //为yi
        if ((w3 dot feature) * label < 1 - thet) {  //对应文献中I2 = {i|yi*w*xi<1-thet} 分类错误的点
          axpy(lambda*(1-thet-label*(feature dot w3))*(1-thet-label*(feature dot w3))/(m*(1-thet)*(1-thet)),feature,g)
        }
        else if ((w3 dot feature) * label > 1 + thet) {  //对应文献中I3 = {i|yi*w*xi > 1 + thet}
          axpy(lambda*mu*(label*(feature dot w3)-thet-1)*(label*(feature dot w3)-thet-1)/ (m * (1 - thet) * (1 - thet)) , feature, g)
        }
      }
      Array(g).iterator
    }.reduce{ (a,b) =>
      axpy(1.0, a, b)//b = a*x+b b=a+b 这里b为gradient
      b
      //      loB = accumulate(b)
    }
    axpy(1.0, gradient, w3)//w = gradient + w
    //    val loB = gradient.head
    w3
  }

  def loss(w2: DenseVector[Double], x: (DenseVector[Double], Int)): Double ={
    // w是权重，x是样本
    val feature = x._1 //为除开标签值的样本
    val label = x._2  //为标签值
    var l = 0.0
    //var l = DenseVector.zeros[Double](1)  //损失
    if ((w2 dot feature) * label < 1 - thet) {
      val g = lambda*(1-thet-label*(feature dot w2))*(1-thet-label*(feature dot w2))/((1-thet)*(1-thet))
      //val loss = lambda*(label*(feature dot w2)+thet-1)/(m*(1-thet)*(1-thet))
      l = l+g+0.5*norm(w2)
      //l = sum(loss,l)
    }
    else if ((w2 dot feature) * label > 1 + thet) {
      val g = lambda*mu*(label*(feature dot w2)-thet-1)*(label*(feature dot w2)-thet-1)/((1-thet)*(1-thet))
      //val loss = lambda*mu*(label*(feature dot w2)-thet-1)/(m*(1-thet)*(1-thet))
      l = l+g+0.5*norm(w2)
      //l = sum(loss,l)
    }
    l
  }

  def gradient(w2: DenseVector[Double], x: (DenseVector[Double], Int)): DenseVector[Double] ={
    // w是权重，x是样本
    val feature = x._1 //为除开标签值的样本
    val label = x._2  //为标签值
    //var l = DenseVector.zeros[Double](1)  //损失
    if ((w2 dot feature) * label < 1 - thet) {
      val g = (2*lambda*(label*(feature dot w2)+thet-1)/((1-thet)*(1-thet)))*label.toDouble * feature
      //val loss = lambda*(label*(feature dot w2)+thet-1)/(m*(1-thet)*(1-thet))  单个点xy不用除m
      axpy(1.0, g, w2)
      //l = sum(loss,l)
    }
    else if ((w2 dot feature) * label > 1 + thet) {
      val g = (2*lambda*mu*(label*(feature dot w2)-thet-1)/((1-thet)*(1-thet)))*label.toDouble * feature
      //val loss = lambda*mu*(label*(feature dot w2)-thet-1)/(m*(1-thet)*(1-thet))
      axpy(1.0, g, w2)
      //l = sum(loss,l)
    }
    w2
  }

  def testgradient(data: RDD[(DenseVector[Double], Int)]): DenseVector[Double] ={
    var w = DenseVector.zeros[Double](n)
    gradient(w, data)
    //val vec = data.take(1)
    //gradient(w, vec(0))
  }

  def svrgTrain(data: RDD[(DenseVector[Double], Int)], K: Int, T: Int, batchSpilt: Double, p: Int, gama:Double): DenseVector[Double] ={
    //batchSpilt在0-1之间，表示采样的比例
    //p表示DSVRG的pi
    var w = DenseVector.rand[Double](n)
    //var w = DenseVector(Array(-100.0, -100.0))
    var detaw = 1.0
    var i = 0
    while (detaw >= exps && i <= T) {
      i = i+1
      val batchData = data.sample(false, batchSpilt).cache()//分发batch数据 false改为true 有放回采样
      var z = w.copy
      var B = batchData.mapPartitionsWithIndex((idx, iter) => if(idx == 0) iter else Iterator())
        .collect()  //mapPartitionWithIndex能把分区的index传递给用户指定的输入函数
      //选取第idx号机器上所有数据
      //B相当于B0-Bpi
      var size = B.length/p  //把batchdata继续分为p份，每一份包含size个数据
      var start = 0
      var index = 0
      val fullloss = lossB(w.copy, batchData)
      //      var z0 = DenseVector.zeros[Double](n)
      //      val z0 = w
      //      val x = w
      for (j <- 1 to K) {
        val x = w.copy
        var z0 = DenseVector.zeros[Double](n)
        val fullGradient = gradient(z.copy, batchData)//w改为z
        for (u <- 0 until size) {
          val sample = B(u + start)
          val gradx = gradient(x.copy, sample)
          //          val l = z
          val gradz = gradient(z.copy, sample)
          val lossx = loss(x.copy, sample)
          val lossz = loss(z.copy, sample)
          axpy(-eta, gradx-gradz+fullGradient+gama*(x-w), x)//改为-eta
          val lossx2 = loss(x.copy, sample)
          axpy(1.0,x,z0)
          //          println(try{
          //            val out = new FileWriter("test8.txt",true)
          //            for(x <- z0) {
          //              out.write(x.toString+' ')
          //            }
          //            out.write('\n')
          //            out.write("-----")
          //            out.write('\n')
          //            out.close()
          //          })
        }
        z = z0/size.toDouble

        println(try{
          val out = new FileWriter("test3.txt",true)
          for(t <- z) {
            out.write(t.toString+' ')
          }
          out.write('\n')
          out.write("+++++++++")
          out.write(fullloss.toString)
          out.write('\n')
          out.close()
        })


        start = start+size
        if (start+size >= B.length) {
          index = index+1
          start = 0
          B = batchData.mapPartitionsWithIndex((idx, iter) => if(idx == index) iter else Iterator())
            .collect()
          size = B.length/p
          if (size == 0) {
            index = 0
            B = batchData.mapPartitionsWithIndex((idx, iter) => if(idx == index) iter else Iterator())
              .collect()
            size = B.length/p
          }
        }
        //        start = start+size
      }
      detaw = norm(z-w)//2范数

      w = z


    }

    w
  }

  def dsvrgWithKmeans(data: RDD[(DenseVector[Double], Int)], K: Int, T: Int): DenseVector[Double] ={
    val start_time = System.currentTimeMillis()
    val sc = data.sparkContext
    val timePath = new Path("hdfs://node18:9000/user/wangyilin/svrg_time.txt")
    val timeConf = new Configuration(sc.hadoopConfiguration)
    val timeFS = timePath.getFileSystem(timeConf)
    val timeWrite = new BufferedOutputStream(timeFS.create(timePath))

    var w = DenseVector.rand[Double](n)
    val data1 = data.map(v=>(v._1*v._2.toDouble, 0))
    val k = 20
    val centers = kmeans(data1, k, 30, 1e-6, 0.5)
    var bcCenters = sc.broadcast(centers)

    val dataWithCluster = data.mapPartitions{ points =>
      val thisCenters = bcCenters.value
      val dataArr = ArrayBuffer[(Int, (DenseVector[Double], Int))]()
      //dataArr记录最近的聚类中心和样本信息
      points.foreach{ v =>
        val loc = v._1 * v._2.toDouble
        var bestDis = Double.PositiveInfinity
        var bestIndex = 0
        var i = 0
        thisCenters.foreach{ center =>
          val dis = sqrt(kernel(loc,loc)+kernel(center,center)-2*kernel(loc,center))
          if (dis < bestDis) {
            bestDis = dis
            bestIndex = i
          }
          i += 1
        }
        val value = (bestIndex, v)
        dataArr += value
      }
      dataArr.iterator
    }

    val Rcollect = ArrayBuffer[(DenseVector[Double], Int)]()

    for (i <- 0 until k) {
      val cluster = dataWithCluster.filter(v => v._1 == i).map(_._2)
      val num = scala.math.ceil(cluster.count()*T*K/this.m.toDouble).toInt
      val Ri = cluster.takeSample(true, num)
      Rcollect ++= Ri
    }
    val R = Random.shuffle(Rcollect.toSeq).toArray

    var start = 0
    for (i <- 1 to K) {
      val fullgradient = gradient(w.copy, data)
      val w1 = w.copy
      var w0 = DenseVector.zeros[Double](n)
      for (j <- 0 until T) {
        val gradient0 = gradient(w1.copy, R(start+j))
        val gradient1 = gradient(w.copy, R(start+j))
        axpy(-eta, gradient0-gradient1+fullgradient, w1)
        w0 = (w1+j.toDouble*w0)/(j+1.0)
      }
      start = start + T
      w = w0.copy
    }
    val time = System.currentTimeMillis() - start_time
    val str = time.toString + '\n'
    timeWrite.write(str.getBytes("UTF-8"))
    timeWrite.flush()
    timeWrite.close()
    timeFS.close()
    w
  }

  def dsvrgTrain(data: RDD[(DenseVector[Double], Int)], K: Int, T: Int): DenseVector[Double] ={
    var w = DenseVector.rand[Double](n)
    val R = data.takeSample(true, T*K)
    val sc = data.sparkContext
    var start = 0
    for (i <- 0 until K) {
      val fullgradient = gradient(w.copy, data)
      val w1 = w.copy
      var w0 = DenseVector.zeros[Double](n)
      for (j <- 0 until T) {
        val gradient0 = gradient(w1.copy, R(start+j))
        val gradient1 = gradient(w.copy, R(start+j))
        axpy(-eta, gradient0-gradient1+fullgradient, w1)
        w0 = (w1+j.toDouble*w0)/(j+1.0)
      }
      start = start + T
      w = w0.copy
    }
    w
  }

  private val kernel= (x: DenseVector[Double], y:DenseVector[Double]) => x dot y
  private val expkernel = (x: DenseVector[Double], y:DenseVector[Double]) => exp(-((x-y)dot(x-y))/2)

  //private val kernal = (x: DenseVector[Double], y:DenseVector[Double], sigma:Double) => exp(-((x-y)dot(x-y))/(2*sigma))

  def trainDualSVM(data: RDD[(((DenseVector[Double], Int),Long),Double,Double,Int)], T: Int): RDD[(((DenseVector[Double], Int),Long),Double,Double,Int)] = {
    val bc = data.sparkContext.broadcast(T)
    val newData = data.mapPartitions{ iter =>
      val duplicateIter = iter.duplicate
      val duplicateIter0 = duplicateIter._2.zipWithIndex.duplicate
      val iter0 = duplicateIter0._1
      val dataWithIndex = duplicateIter0._2.toArray
      val dataWithIndex2 = ArrayBuffer[((((DenseVector[Double], Int),Long),Double,Double,Int),Int)]()
      dataWithIndex2 ++= dataWithIndex
      val row = dataWithIndex.length
      val H = DenseMatrix.zeros[Double](2*row,2*row)
      val T = bc.value
      val alpha = DenseVector.zeros[Double](2*row)
      //val m = row
      iter0.foreach{ v =>
        val x0 = v._1._1._1._1
        val y0 = v._1._1._1._2
        val index0 = v._2
        alpha.update(index0, v._1._2)
        alpha.update(index0+row, v._1._3)
        dataWithIndex2.foreach { v1 =>
          val x1 = v1._1._1._1._1
          val y1 = v1._1._1._1._2
          val index1 = v1._2
          val value = kernel(x0,x1)*y0*y1
          if (index1 != index0) {
            H.update(index1, index0, value)
            H.update(index0, index1, value)
            H.update(index1+row, index0, -value)
            H.update(index0+row, index1, -value)
            H.update(index1, index0+row, -value)
            H.update(index0, index1+row, -value)
            H.update(index1+row, index0+row, value)
            H.update(index0+row, index1+row, value)
          }
          else {
            H.update(index1+row, index1, -value)
            H.update(index1, index1+row, -value)
            H.update(index1, index1, value+m*(1-thet)*(1-thet)/(2*lambda))
            H.update(index1+row, index1+row, value+m*(1-thet)*(1-thet)/(2*lambda*mu))
          }
        }
        dataWithIndex2.remove(0)
      }
      //val alpha = DenseVector.zeros[Double](2*row)
      var detaAlpha = 1.0
      var j = 0
      var oldAlpha = alpha.copy
      while (detaAlpha >= exps && j <= T) {
        j = j + 1
        for (i <- 0 until row) {
          val gradient = (H(i,::).t dot alpha) + thet - 1.0
          val newAlpha = alpha(i) - gradient/H(i,i)
          if (newAlpha < 0)
            alpha.update(i, 0.0)
          else
            alpha.update(i, newAlpha)
        }
        for (i <- row until 2*row) {
          val gradient = (H(i,::).t dot alpha) + thet + 1.0
          val newAlpha = alpha(i) - gradient/H(i,i)
          if (newAlpha < 0)
            alpha.update(i, 0.0)
          else
            alpha.update(i, newAlpha)
        }
        detaAlpha = norm(alpha-oldAlpha)
        oldAlpha = alpha.copy
      }
      val dataUpdate = ArrayBuffer[(((DenseVector[Double], Int),Long),Double,Double,Int)]()
      for (i <- 0 until row) {
        val newdata = (dataWithIndex(i)._1._1, alpha(i), alpha(i+row), dataWithIndex(i)._2)
        dataUpdate += newdata
      }
      dataUpdate.iterator
    }
    newData
  }

  def trainDualSVMWithVector(data: RDD[(((DenseVector[Double], Int),Long),Double,Double,Int)], T: Int): RDD[(((DenseVector[Double], Int),Long),Double,Double,Int)] = {
    val bc = data.sparkContext.broadcast(T)
    val newData = data.mapPartitions{ iter =>
      val duplicateIter = iter.duplicate
      val duplicateIter0 = duplicateIter._2.zipWithIndex.duplicate
      val iter0 = duplicateIter0._1
      val dataWithIndex = duplicateIter0._2.toArray
      val row = dataWithIndex.length
      val T = bc.value
      val alpha = DenseVector.zeros[Double](2*row)

      iter0.foreach{ v =>
        val index0 = v._2
        alpha.update(index0, v._1._2)
        alpha.update(index0+row, v._1._3)
      }

      //val alpha = DenseVector.zeros[Double](2*row)
      var detaAlpha = 1.0
      var j = 0
      var oldAlpha = alpha.copy
      while (detaAlpha >= exps && j <= T) {
        j += 1
        for (i <- 0 until row) {
          val Hi = DenseVector.zeros[Double](2*row)
          val vi = dataWithIndex(i)._1._1._1._1
          val yi = dataWithIndex(i)._1._1._1._2

          for (p <- 0 until row) {
            val vp = dataWithIndex(p)._1._1._1._1
            val yp = dataWithIndex(p)._1._1._1._2
            Hi.update(p, kernel(vi, vp)*yi*yp)
            Hi.update(p + row, -kernel(vi,vp)*yi*yp)
          }
          val Hii = kernel(vi, vi) + row*(1-thet)*(1-thet)/(2*lambda)
          Hi.update(i, Hii)

          val gradient = (Hi dot alpha) + thet - 1.0
          val newAlpha = alpha(i) - gradient/Hii
          if (newAlpha < 0)
            alpha.update(i, 0.0)
          else
            alpha.update(i, newAlpha)
        }

        for (i <- 0 until row) {

          val Hi = DenseVector.zeros[Double](2*row)
          val vi = dataWithIndex(i)._1._1._1._1
          val yi = dataWithIndex(i)._1._1._1._2

          for (p <- 0 until row) {
            val vp = dataWithIndex(p)._1._1._1._1
            val yp = dataWithIndex(p)._1._1._1._2
            Hi.update(p, -kernel(vi, vp)*yi*yp)
            Hi.update(p + row, kernel(vi,vp)*yi*yp)
          }
          val Hii = kernel(vi, vi) + row*(1-thet)*(1-thet)/(2*lambda*mu)
          Hi.update(i + row, Hii)

          val gradient = (Hi dot alpha) + thet + 1.0
          val newAlpha = alpha(i) - gradient/Hii
          if (newAlpha < 0)
            alpha.update(i + row, 0.0)
          else
            alpha.update(i + row, newAlpha)
        }
        detaAlpha = norm(alpha-oldAlpha)
        oldAlpha = alpha.copy
      }
      val dataUpdate = ArrayBuffer[(((DenseVector[Double], Int),Long),Double,Double,Int)]()
      for (i <- 0 until row) {
        val newdata = (dataWithIndex(i)._1._1, alpha(i), alpha(i+row), dataWithIndex(i)._2)
        dataUpdate += newdata
      }
      dataUpdate.iterator
    }
    newData
  }

  def getDualResult(data: RDD[(((DenseVector[Double], Int),Long),Double,Double,Int)]): SparseVector[Double] = {
    val v = data.map{ v =>
      val y = v._1._1._2
      (y*(v._2 - v._3),v._1._2)
    }.filter(_._1 != 0.0).collect().sortBy(_._2)
    val V = SparseVector.zeros[Double](m.toInt)
    v.foreach{ v1 =>
      V.update(v1._2.toInt, v1._1)
    }
    V
  }

  def centroid(data: RDD[(DenseVector[Double], Int)], k: Int, split: Double): Array[DenseVector[Double]] = {
    var i = 1
    var converged = false
    val sample = data.sample(false, split)
    val sc = sample.sparkContext
    //val size = sample.count()
    val centers = ArrayBuffer[DenseVector[Double]]()

    centers += sample.map{ v=>
      (kernel(v._1, v._1), v._1)
    }.sortBy(v=>v._1).take(1)(0)._2


    while (i < k) {
      val Q = DenseMatrix.zeros[Double](i,i)
      for (p<-0 until i) {
        for (q<-0 to p) {
          Q.update(p, q, kernel(centers(p), centers(q)))
        }
      }
      val bcQ = sc.broadcast(inv(Q))
      val bcCenters = sc.broadcast(centers.toArray)
      val totalContribs = sample.mapPartitions { iter =>
        val thisQ = bcQ.value
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.length
        val length = thisCenters.length
        val sums = ArrayBuffer[(Double, DenseVector[Double])]()
        iter.foreach{ v=>
          var loc = v._1
          var k_q = DenseMatrix.zeros[Double](length,1)
          for (j <- 0 until length) {
            k_q.update(j, 0, kernel(loc, thisCenters(j)))
          }
          val sc = k_q.t * thisQ * k_q
          sums.append((sc(0, 0), loc))
        }
        sums.toArray.iterator
      }
      centers += totalContribs.sortBy(v=>v._1).take(1)(0)._2
      i += 1
    }
    centers.toArray
  }

  def kmeans(data: RDD[(DenseVector[Double], Int)], k: Int, Iter: Int, epsilon: Double, split: Double): Array[DenseVector[Double]] = {
    var i = 0
    var converged = false
    val sample = data.sample(false, split)
    val centers = sample.map(_._1).take(k)
    val sc = sample.sparkContext
    //val size = sample.count()

    while (i < Iter && !converged) {
      val bcCenters = sc.broadcast(centers)
      val totalContribs = sample.mapPartitions { iter =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.length
        val sums = Array.fill(thisCenters.length)(DenseVector.zeros[Double](dims))
        val counts = Array.fill(thisCenters.length)(0L)
        iter.foreach{ v =>
          val loc = v._1
          var bestDis = Double.PositiveInfinity
          var bestIndex = 0
          var i = 0
          thisCenters.foreach{ center =>
            val dis = sqrt(kernel(loc,loc)+kernel(center,center)-2*kernel(loc,center))
            if (dis < bestDis) {
              bestDis = dis
              bestIndex = i
            }
            i += 1
          }
          axpy(1.0, loc, sums(bestIndex))
          counts(bestIndex) += 1
        }
        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>
        axpy(1.0, sum2, sum1)
        (sum1, count1 + count2)
      }.collectAsMap()
      bcCenters.destroy()
      converged = true
      totalContribs.foreach { case (j, (sum, count)) =>
        val newCenter = sum/count.toDouble
        if (converged && norm(newCenter - centers(j)) > epsilon * epsilon) {
          converged = false
        }
        centers(j) = newCenter
      }
      i += 1
    }
    centers

    //val flag = centers.distinct
    /*
    val bcCenters = sc.broadcast(centers)
    val dataWithPartition = data.mapPartitions{ points =>
      val thisCenters = bcCenters.value
      val dataArr = ArrayBuffer[(Int, ((DenseVector[Double], Int)))]()
      points.foreach{ v =>
        val loc = v._1
        var bestDis = Double.PositiveInfinity
        var bestIndex = 0
        var i = 0
        thisCenters.foreach{ center =>
          val dis = sqrt(norm(loc - center))
          if (dis < bestDis) {
            bestDis = dis
            bestIndex = i
          }
          i += 1
        }
        val value = (bestIndex, v)
        dataArr += value
      }
      dataArr.iterator
    }.partitionBy(new SparkPartitionByCenter(k)).map(_._2)
    dataWithPartition.cache()
    */
  }

  def DCSolver(data: RDD[(DenseVector[Double], Int)], T: Int, L: Int, split: Double): SparseVector[Double] = {
    val start_time = System.currentTimeMillis()
    var l = L
    var dataWithAlpha = data.zipWithIndex().map(v => (v,0.0,0.0,0))
    var sv = data
    val sc = dataWithAlpha.sparkContext

    val timePath = new Path("hdfs://node18:9000/user/wangyilin/dc_time.txt")
    val timeConf = new Configuration(sc.hadoopConfiguration)
    val timeFS = timePath.getFileSystem(timeConf)
    val timeWrite = new BufferedOutputStream(timeFS.create(timePath))

    val vPath = new Path("hdfs://node18:9000/user/wangyilin/dc_v.txt")
    val vConf = new Configuration(sc.hadoopConfiguration)
    val vFS = timePath.getFileSystem(vConf)
    val vWrite = new BufferedOutputStream(vFS.create(vPath))

    //var size = 0L
    while (l >= 1) {
      val k = pow(3, l)
      val centers = kmeans(sv, k, 30, 1e-6, split)
      /*
      val size = dataRePartition.mapPartitions{ iter =>
        val arr = new Array[Int](1)
        arr(0) = iter.length
        arr.iterator
      }.collect().sorted.reverse
      */
      /*
      val size = dataRePartition.mapPartitions{ iter =>
        val size = iter.length
        Array(size).iterator
      }.collect().sorted.reverse
      */

      val bcCenters = sc.broadcast(centers)
      val dataWithPartition = dataWithAlpha.mapPartitions{ points =>
        val thisCenters = bcCenters.value
        val dataArr = ArrayBuffer[(Int, (((DenseVector[Double], Int),Long),Double,Double,Int))]()
        //dataArr记录最近的聚类中心和样本信息
        points.foreach{ v =>
          val loc = v._1._1._1//样本位置
          var bestDis = Double.PositiveInfinity
          var bestIndex = 0
          var i = 0
          thisCenters.foreach{ center =>
            val dis = sqrt(norm(loc - center))
            if (dis < bestDis) {
              bestDis = dis
              bestIndex = i
            }
            i += 1
          }
          val value = (bestIndex, v)
          dataArr += value
        }
        dataArr.iterator
      }.partitionBy(new SparkPartitionByCenter(k)).map(_._2).mapPartitions{ cluster=>
        var index = 0
        val dataArr = ArrayBuffer[(((DenseVector[Double], Int),Long),Double,Double,Int)]()
        cluster.foreach{ point =>
          val newpoint = (point._1,point._2,point._3,index)
          dataArr += newpoint
          index += 1
        }
        dataArr.iterator
      }.cache()

      //dataWithAlpha = trainDualSVM(dataWithPartition, T)
      dataWithAlpha = trainDualSVM(dataWithPartition, T)

      val time = System.currentTimeMillis() - start_time
      val str = time.toString + '\n'
      timeWrite.write(str.getBytes("UTF-8"))
      timeWrite.flush()

      val V = getDualResult(dataWithAlpha)
      for(x <- V) {
        val str = x.toString+','
        vWrite.write(str.getBytes("UTF-8"))
      }
      val st = "\n"
      vWrite.write(st.getBytes("UTF-8"))
      vWrite.flush()


      //val size = sv.count()
      sv = dataWithAlpha.map(_._1._1).cache()


      l -= 1
    }

    val dataWithPartition = dataWithAlpha.filter{ v =>
      v._2 + v._3 > 0.0
    }.repartition(1).cache()
    dataWithAlpha = trainDualSVM(dataWithPartition, T)

    val time = System.currentTimeMillis() - start_time
    val str = time.toString + '\n'
    timeWrite.write(str.getBytes("UTF-8"))
    timeWrite.flush()

    val V = getDualResult(dataWithAlpha)
    for(x <- V) {
      val str = x.toString+','
      vWrite.write(str.getBytes("UTF-8"))
    }
    val st = "\n"
    vWrite.write(st.getBytes("UTF-8"))
    vWrite.flush()

    timeWrite.close()
    timeFS.close()

    vFS.close()
    vFS.close()

    V
  }

  def RandomSolver(data: RDD[(DenseVector[Double], Int)], T: Int, L: Int): SparseVector[Double] = {
    val start_time = System.currentTimeMillis()
    var l = L
    var dataWithAlpha = data.zipWithIndex().repartition(pow(3,l)).map(v => (v,0.0,0.0,TaskContext.get.partitionId)).cache()

    val sc = dataWithAlpha.sparkContext
    val timePath = new Path("hdfs://node18:9000/user/wangyilin/random_time.txt")
    val timeConf = new Configuration(sc.hadoopConfiguration)
    val timeFS = timePath.getFileSystem(timeConf)
    val timeWrite = new BufferedOutputStream(timeFS.create(timePath))

    val vPath = new Path("hdfs://node18:9000/user/wangyilin/random_v.txt")
    val vConf = new Configuration(sc.hadoopConfiguration)
    val vFS = timePath.getFileSystem(vConf)
    val vWrite = new BufferedOutputStream(vFS.create(vPath))

    while (l >= 1) {
      val dataAfterDualTrain = trainDualSVM(dataWithAlpha, T)

      val time = System.currentTimeMillis() - start_time
      val str = time.toString + '\n'
      timeWrite.write(str.getBytes("UTF-8"))
      timeWrite.flush()

      val V = getDualResult(dataAfterDualTrain)
      for(x <- V) {
        val str = x.toString+','
        vWrite.write(str.getBytes("UTF-8"))
      }
      val st = "\n"
      vWrite.write(st.getBytes("UTF-8"))
      vWrite.flush()

      l -= 1
      if (l >= 1) {
        val bck = pow(3,l)
        dataWithAlpha = dataAfterDualTrain.map{
          v => ((v._4 % bck), v)
        }.partitionBy(new SparkPartitionByCenter(bck)).map(_._2).cache()
      }

    }
    timeWrite.close()
    timeFS.close()

    vFS.close()
    vFS.close()

    val V = getDualResult(dataWithAlpha)
    V
  }

  def CascadeSolver(data: RDD[(DenseVector[Double], Int)], T: Int, L: Int): SparseVector[Double] = {
    val start_time = System.currentTimeMillis()
    var l = L
    var dataWithAlpha = data.zipWithIndex().repartition(pow(2,l)).map(v => (v,0.0,0.0,TaskContext.get.partitionId)).cache()

    val sc = dataWithAlpha.sparkContext
    val timePath = new Path("hdfs://node18:9000/user/wangyilin/cascade_time.txt")
    val timeConf = new Configuration(sc.hadoopConfiguration)
    val timeFS = timePath.getFileSystem(timeConf)
    val timeWrite = new BufferedOutputStream(timeFS.create(timePath))

    val vPath = new Path("hdfs://node18:9000/user/wangyilin/cascade_v.txt")
    val vConf = new Configuration(sc.hadoopConfiguration)
    val vFS = timePath.getFileSystem(vConf)
    val vWrite = new BufferedOutputStream(vFS.create(vPath))

    while (l >= 0) {
      val dataAfterDualTrain = trainDualSVM(dataWithAlpha, T)
      val time = System.currentTimeMillis() - start_time
      val str = time.toString + '\n'
      timeWrite.write(str.getBytes("UTF-8"))
      timeWrite.flush()

      val V = getDualResult(dataAfterDualTrain)
      for(x <- V) {
        val str = x.toString+','
        vWrite.write(str.getBytes("UTF-8"))
      }
      val st = "\n"
      vWrite.write(st.getBytes("UTF-8"))
      vWrite.flush()

      l -= 1
      if (l >= 0) {
        val bck = pow(2,l)
        dataWithAlpha = dataAfterDualTrain.filter(v => v._2 + v._3 != 0.0).map{
          v => ((v._4 % bck), v)
        }.partitionBy(new SparkPartitionByCenter(bck)).map(_._2).cache()
      }

    }

    timeWrite.close()
    timeFS.close()

    vFS.close()
    vFS.close()

    val V = getDualResult(dataWithAlpha)
    V
  }

  def DPPSolver(data: RDD[(DenseVector[Double], Int)], T: Int, L: Int, split: Double, numCluster: Int): SparseVector[Double] = {
    val start_time = System.currentTimeMillis()
    var l = L
    var dataWithAlpha = data.zipWithIndex().map(v => (v,0.0,0.0,0))
    val sc = dataWithAlpha.sparkContext
    val centers = kmeans(data, numCluster, 50, 1e-6, split)
    val bcCenters = sc.broadcast(centers)

    val timePath = new Path("hdfs://node18:9000/user/wangyilin/dpp_time.txt")
    val timeConf = new Configuration(sc.hadoopConfiguration)
    val timeFS = timePath.getFileSystem(timeConf)
    val timeWrite = new BufferedOutputStream(timeFS.create(timePath))

    val vPath = new Path("hdfs://node18:9000/user/wangyilin/dpp_v.txt")
    val vConf = new Configuration(sc.hadoopConfiguration)
    val vFS = timePath.getFileSystem(vConf)
    val vWrite = new BufferedOutputStream(vFS.create(vPath))

    val bck = pow(2,l)
    val dataWithPartition = dataWithAlpha.mapPartitions{ points =>
      val thisCenters = bcCenters.value
      val dataArr = ArrayBuffer[(Int, (((DenseVector[Double], Int),Long),Double,Double,Int))]()
      //dataArr记录最近的聚类中心和样本信息
      points.foreach{ v =>
        val loc = v._1._1._1//样本位置
      var bestDis = Double.PositiveInfinity
        var bestIndex = 0
        var i = 0
        thisCenters.foreach{ center =>
          val dis = sqrt(kernel(loc,loc)+kernel(center,center)-2*kernel(loc,center))
          if (dis < bestDis) {
            bestDis = dis
            bestIndex = i
          }
          i += 1
        }
        val value = (bestIndex, v)
        dataArr += value
      }
      dataArr.iterator
    }.partitionBy(new SparkPartitionByCenter(numCluster)).mapPartitions{
      iter =>
        val k = bck
        var i = 0
        val dataArr = ArrayBuffer[(Int, (((DenseVector[Double], Int),Long),Double,Double,Int))]()
        iter.foreach{ v =>
          val value = (i, (v._2._1, v._2._2, v._2._3, i))
          dataArr += value
          i = (i + 1) % k
        }
        dataArr.iterator
    }.partitionBy(new SparkPartitionByCenter(bck)).cache()
    dataWithAlpha = dataWithPartition.map(v=>v._2)

    while (l >= 0) {
      val dataAfterDualTrain = trainDualSVM(dataWithAlpha, T)

      val time = System.currentTimeMillis() - start_time
      val str = time.toString + '\n'
      timeWrite.write(str.getBytes("UTF-8"))
      timeWrite.flush()

      val V = getDualResult(dataAfterDualTrain)
      for(x <- V) {
        val str = x.toString+','
        vWrite.write(str.getBytes("UTF-8"))
      }
      val st = "\n"
      vWrite.write(st.getBytes("UTF-8"))
      vWrite.flush()

      l -= 1
      if (l >= 0) {
        val bck = pow(2,l)
        dataWithAlpha = dataAfterDualTrain.filter(v => v._2 + v._3 != 0.0).map{
          v => ((v._4 % bck), v)
        }.partitionBy(new SparkPartitionByCenter(bck)).map(_._2)
      }
    }
    timeWrite.close()
    timeFS.close()

    vFS.close()
    vFS.close()

    //dataWithAlpha = trainDualSVM(dataWithAlpha, T)
    val V = getDualResult(dataWithAlpha)
    V
  }

  def KmeansSolver(data: RDD[(DenseVector[Double], Int)], T: Int, L: Int, split: Double, numCluster: Int): SparseVector[Double] = {
    val start_time = System.currentTimeMillis()

    var l = L
    var dataWithAlpha = data.zipWithIndex().map(v => (v,0.0,0.0,0))
    val sc = dataWithAlpha.sparkContext
    val centers = centroid(data, 50, split)
    val bcCenters = sc.broadcast(centers)
    val bck = pow(2,l)

    val timePath = new Path("di_time.txt")
    val timeConf = new Configuration(sc.hadoopConfiguration)
    val timeFS = timePath.getFileSystem(timeConf)
    val timeWrite = new BufferedOutputStream(timeFS.create(timePath))

    val vPath = new Path("di_v.txt")
    val vConf = new Configuration(sc.hadoopConfiguration)
    val vFS = timePath.getFileSystem(vConf)
    val vWrite = new BufferedOutputStream(vFS.create(vPath))

    val dataWithPartition = dataWithAlpha.mapPartitions{ points =>
      val thisCenters = bcCenters.value
      val dataArr = ArrayBuffer[(Int, (((DenseVector[Double], Int),Long),Double,Double,Int))]()
      //dataArr记录最近的聚类中心和样本信息
      points.foreach{ v =>
        val loc = v._1._1._1//样本位置
        var bestDis = Double.PositiveInfinity
        var bestIndex = 0
        var i = 0
        thisCenters.foreach{ center =>
          val dis = sqrt(expkernel(loc,loc)+expkernel(center,center)-2*expkernel(loc,center))
          if (dis < bestDis) {
            bestDis = dis
            bestIndex = i
          }
          i += 1
        }
        val value = (bestIndex, v)
        dataArr += value
      }
      dataArr.iterator
    }.partitionBy(new SparkPartitionByCenter(numCluster)).mapPartitions{
      iter =>
        val k = bck
        var i = 0
        val dataArr = ArrayBuffer[(Int, (((DenseVector[Double], Int),Long),Double,Double,Int))]()
        iter.foreach{ v =>
          val value = (i, (v._2._1, v._2._2, v._2._3, i))
          dataArr += value
          i = (i + 1) % k
        }
        dataArr.iterator
    }.partitionBy(new SparkPartitionByCenter(bck)).cache()
    dataWithAlpha = dataWithPartition.map(v=>v._2)

    while (l >= 2) {
      val dataAfterDualTrain = trainDualSVM(dataWithAlpha, T)
      val time = System.currentTimeMillis() - start_time
      val str = time.toString + '\n'
      timeWrite.write(str.getBytes("UTF-8"))
      timeWrite.flush()

      val V = getDualResult(dataAfterDualTrain)
      for(x <- V) {
        val str = x.toString+','
        vWrite.write(str.getBytes("UTF-8"))
      }
      val st = "\n"
      vWrite.write(st.getBytes("UTF-8"))
      vWrite.flush()

      l -= 1
      val bck = pow(2,l)

      if (l >= 2) {
        dataWithAlpha = dataAfterDualTrain.map{
          v => ((v._4 % bck), v)
        }.partitionBy(new SparkPartitionByCenter(bck)).map(_._2)
      }

    }
    timeWrite.close()
    timeFS.close()

    vFS.close()
    vFS.close()

    val V = SparseVector.zeros[Double](1)
    V
  }

}

