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
$ YOUR_SPARK_HOME/bin/spark-submit \
  --class "ODMTest" \
  target/scala-2.12/simple-project_2.12-1.0.jar
```

## Authors and Copyright

SODM is developed in National Engineering Research Center for Big Data Technology and System, Cluster and Grid Computing Lab, Services Computing Technology and System Lab, School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China by Yilin Wang (yilin_wang@hust.edu.cn), Nan Cao(nan_cao@hust.edu.cn), Teng Zhang(tengzhang@hust.edu.cn), Xuanhua Shi (xhshi@hust.edu.cn), Hai Jin (hjin@hust.edu.cn).

Copyright (C) 2021, [STCS & CGCL](http://grid.hust.edu.cn/) and [Huazhong University of Science and Technology](https://www.hust.edu.cn/).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```
  http://www.apache.org/licenses/LICENSE-2.0
```
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Publications
Yilin Wang, Nan Cao, Teng Zhang, Xuanhua Shi, Hai Jin, "Scalable Optimal Margin Distribution Machine", in Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI-23), Macao, China, August 19th-25th, 2023
