package ODMModel
import org.apache.spark.Partitioner

class SparkPartitionByCenter(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts

  override def getPartition(key: Any): Int = {
    key match {
      case k: Int => k%numParts
      case _ => 0
    }
  }
  override def equals(other: Any): Boolean = other match {
    case mypartition: SparkPartitionByCenter =>
      mypartition.numPartitions == numPartitions
    case _ =>
      false
  }
  override def hashCode: Int = numPartitions
}
