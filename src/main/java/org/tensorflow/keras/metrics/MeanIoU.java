/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.tf.ConfusionMatrix;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat64;

/**
 * Computes the mean Intersection-Over-Union metric. 
 * 
 * Mean Intersection-Over-Union is a common evaluation metric for semantic image
 * segmentation, which first computes the IOU for each semantic class and then
 * computes the average over classes. IOU is defined as follows:
 * <p>  IOU = true_positive / (true_positive + false_positive + false_negative).
 * 
 * @author jbclarke
 */
public class MeanIoU extends Metric {
    public static final String TOTAL_CONFUSION_MATRIX = "TOTAL_CONFUSION_MATRIX";
    private Variable<TFloat64> totalCM;
    /**
     * The possible number of labels the prediction task can have. 
     * This value must be provided, since a confusion matrix of 
     * dimension = [num_classes, num_classes] will be allocated.
     */
    private final long numClasses;
    
    /**
     * create a metric with name = class name and reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param numClasses The possible number of labels the prediction task can have
     */
    protected MeanIoU(Ops tf, long numClasses) {
        this(tf, null, numClasses,  null);
    }

    /**
     * create a metric with reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     * @param numClasses The possible number of labels the prediction task can have
     */
    protected MeanIoU(Ops tf, String name, long numClasses) {
        this(tf, name, numClasses, null);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param numClasses The possible number of labels the prediction task can have
     * @param dType the DataType to use
     */
    protected MeanIoU(Ops tf,  long numClasses, DataType dType) {
        this(tf, null, numClasses, dType);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow ops
     * @param name the name of this metric
     * @param numClasses The possible number of labels the prediction task can have
     * @param dType the DataType
     */
    protected MeanIoU(Ops tf, String name, long numClasses, DataType dType) {
        super(tf, name, dType);
        this.numClasses = numClasses;
        init();
    }
    
    private void init() {
        Zeros zeros = new Zeros(tf);
        
        this.totalCM = getVariable(TOTAL_CONFUSION_MATRIX);
        if (this.getTotalCM() == null) {
            
            this.totalCM = tf.withName(TOTAL_CONFUSION_MATRIX).variable(zeros.call(tf.constant(Shape.of(this.getNumClasses(), this.getNumClasses())), TFloat64.DTYPE));
            this.addVariable(TOTAL_CONFUSION_MATRIX, this.getTotalCM(), zeros);
        }
    }
 
    /**
     * {@inheritDoc}
     */
    @Override
    public Op updateState(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeight = args.length > 2 ? args[2] : null;
        
        yTrue = tf.shape.flatten(tf.dtypes.cast(yTrue, this.dType));
        yPred = tf.shape.flatten(tf.dtypes.cast(yPred, this.dType));
        
        if(sampleWeight != null) {
            sampleWeight = tf.shape.flatten(tf.dtypes.cast(sampleWeight, this.dType));
        }
        
        Operand currentCM = ConfusionMatrix.confusionMatrix(tf, yTrue, yPred, 
                tf.constant(this.getNumClasses()),
                sampleWeight, TFloat64.DTYPE);
        return tf.assignAdd(this.getTotalCM(), currentCM);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand result() {
        Operand sumOverRow = tf.dtypes.cast(tf.reduceSum(this.getTotalCM(), tf.constant(0)), this.dType);
        Operand sumOverCol = tf.dtypes.cast(tf.reduceSum(this.getTotalCM(), tf.constant(1)), this.dType);
        Operand truePositives = tf.dtypes.cast(tf.linalg.matrixDiagPart(getTotalCM(), tf.constant(0), 
                        tf.dtypes.cast(tf.constant(0), this.getTotalCM().asOutput().dataType())),
                this.dType);
        Operand denomintor = tf.math.add(sumOverRow, tf.math.sub(sumOverCol, truePositives));
        Operand numValidEntries = tf.reduceSum(
           tf.dtypes.cast(    
                tf.math.notEqual(denomintor, tf.dtypes.cast(tf.constant(0), denomintor.asOutput().dataType())),
                this.dType), K.allAxis(tf, denomintor));
        Operand iou = tf.math.divNoNan(truePositives, denomintor);
        
        Operand iouSum = tf.reduceSum(iou, K.allAxis(tf, iou));
        return tf.math.divNoNan(iouSum, numValidEntries);
    }

    /**
     * @return the totalCM
     */
    public Variable<TFloat64> getTotalCM() {
        return totalCM;
    }

    /**
     * @return the numClasses
     */
    public long getNumClasses() {
        return numClasses;
    }
    
    
}
