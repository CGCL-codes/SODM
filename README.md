# SODM
ODM is a newly proposed statistical learning framework rooting in the latest margin theory, which demonstrates better generalization performance than the traditional large margin based counterparts. However, it suffers from the ubiquitous scalability problem regarding both computation time and memory storage as other kernel methods. This paper proposes a scalable ODM, which can achieve nearly ten times speedup compared to the original ODM training method. For nonlinear kernels, we put forward a novel distribution-aware partition method to make the local ODM trained on each partition be close and converge fast to the global one. When linear kernel is applied, we extend a communication efficient SVRG method to accelerate the training further. Extensive empirical studies validate that our proposed method is highly computational efficient and almost never worsen the generalization.

## How to use?
We perform the experiments on a Spark cluster with one master and five workers. Each machine is equipped with 16 Intel Xeon E5-2670 CPU cores and 64GB RAM.

Build and run the example

```txt
$ find .
$ sbt package
...
[info] Packaging {..}/{..}/target/scala-2.12/SODM_2.12-1.0.jar
$ $ YOUR_SPARK_HOME/bin/spark-submit \
  --class "SimpleApp" \
  target/scala-2.12/simple-project_2.12-1.0.jar
```
