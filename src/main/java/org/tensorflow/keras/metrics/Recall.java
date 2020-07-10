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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.metrics.impl.ConfusionMatrixEnum;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class Recall extends Metric {
    public static final int NO_CLASSID = -1;
    public static final int NO_TOPK = -1;
    public static final float DEFAULT_THRESHOLD = 0.5f;
    public static final String TRUE_POSITIVES = "TRUE_POSITIVES";
    public static final String FALSE_NEGATIVES = "FALSE_NEGATIVES";
    
    private final float[] thresholds;
    private final Integer topK;
    private final Integer classID;
    
    private Variable<TFloat32> truePositives;
    private Variable<TFloat32> falseNegatives;
    private final String truePositivesName;
    private final String falseNegativesName;

    /**
     * @return the truePositivesName
     */
    public String getTruePositivesName() {
        return truePositivesName;
    }

    /**
     * @return the falseNegativesName
     */
    public String getFalseNegativesName() {
        return falseNegativesName;
    }
    
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     */
    public Recall(Ops tf) {
        this(tf, null, new float[] {DEFAULT_THRESHOLD}, NO_TOPK, NO_CLASSID, null);
    }

    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param threshold A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     */
    public Recall(Ops tf, float threshold) {
        this(tf, null, new float[] { threshold }, NO_TOPK, NO_CLASSID, null);
    }
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param thresholds A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     */
    public Recall(Ops tf, float[] thresholds) {
        this(tf, null, thresholds, NO_TOPK, NO_CLASSID, null);
    }
    
   
    
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param threshold A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     */
    public Recall(Ops tf, String name, float threshold) {
        this(tf, name, new float[] { threshold }, NO_TOPK, NO_TOPK, null);
    }
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param thresholds A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     */
    public Recall(Ops tf, String name, float[] thresholds) {
        this(tf, name, thresholds, NO_TOPK, NO_TOPK, null);
    }
    
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param threshold A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     * @param topK Optional int value specifying the top-k
     *  predictions to consider when calculating recall.
     * Use NO_TOPK for indicating no  topK processing.
     * @param classID Optional int class ID for which we want binary metrics.
     *  This must be in the half-open interval `[0, num_classes)`, where
     *  `num_classes` is the last dimension of predictions. 
     *  Use NO_CLASSID for indicating no classID processing.
     * @param dType the type of the metric result
     */
    public Recall(Ops tf,  float threshold, int topK,
            int classID) {
        this(tf, null, new float[] {threshold}, topK, classID, null);
    }
    
     /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param topK Optional int value specifying the top-k
     *  predictions to consider when calculating recall.
     * Use NO_TOPK for indicating no  topK processing.
     * @param classID Optional int class ID for which we want binary metrics.
     *  This must be in the half-open interval `[0, num_classes)`, where
     *  `num_classes` is the last dimension of predictions. 
     *  Use NO_CLASSID for indicating no classID processing.
     */
    public Recall(Ops tf,  int topK, int classID) {
        this(tf, null, null, topK, classID, null);
    }
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'Recall'
     * @param threshold A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     * @param topK Optional int value specifying the top-k
     *  predictions to consider when calculating recall.
     * @param classID Optional int class ID for which we want binary metrics.
     *  This must be in the half-open interval `[0, num_classes)`, where
     *  `num_classes` is the last dimension of predictions. 
     * @param dType the type of the metric result
     */
    public Recall(Ops tf, String name, float threshold, int topK,
            int classID) {
        this(tf, name, new float[] {threshold}, topK, classID, null);
    }
    
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'Recall'
     * @param thresholds A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     * @param topK Optional int value specifying the top-k
     *  predictions to consider when calculating recall.
     * @param classID Optional Integer class ID for which we want binary metrics.
     *  This must be in the half-open interval `[0, num_classes)`, where
     *  `num_classes` is the last dimension of predictions. 
     */
    public Recall(Ops tf, String name, float[] thresholds, int topK,
            int classID) {
        this(tf, name, thresholds, topK, classID, null);
    }
    
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'Recall'
     * @param threshold A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     * @param topK Optional int value specifying the top-k
     *  predictions to consider when calculating recall.
     * @param classID Optional Integer class ID for which we want binary metrics.
     *  This must be in the half-open interval `[0, num_classes)`, where
     *  `num_classes` is the last dimension of predictions. 
     * @param dType the type of the metric result
     */
    public Recall(Ops tf, String name, float threshold, int topK,
            int classID, DataType dType) {
        this(tf, name, new float[] {threshold}, topK, classID, dType);
    }
    /**
     * Creates a `Recall` metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'Recall'
     * @param thresholds A threshold is compared with prediction
     *  values to determine the truth value of predictions (i.e., above the
     *  threshold is `true`, below is `false`). Default is 0.5
     * @param topK Optional int value specifying the top-k
     *  predictions to consider when calculating recall.
     * @param classID Optional Integer class ID for which we want binary metrics.
     *  This must be in the half-open interval `[0, num_classes)`, where
     *  `num_classes` is the last dimension of predictions. 
     * @param dType the type of the metric result
     */
    public Recall(Ops tf, String name, float[] thresholds, int topK,
            int classID, DataType dType) {
        super(tf, name, dType);
        this.truePositivesName = this.getClass().getSimpleName() + "_" + TRUE_POSITIVES;
        this.falseNegativesName = this.getClass().getSimpleName() + "_" + FALSE_NEGATIVES;
        float defaultThreshold = topK == NO_TOPK ? 0.5f : Metrics.NEG_INF;
        
        this.thresholds =  thresholds == null ? new float[]{defaultThreshold} : thresholds ;
        this.topK = topK == NO_TOPK ? null : topK;
        this.classID = classID == NO_CLASSID ? null : classID;
        
        init();
    }
    
    /**
     * Initialize the Variables
     */
    private void init() {
        Zeros zeros = new Zeros(tf);
        
        this.truePositives = getVariable(truePositivesName);
        if (truePositives == null) {
            
            truePositives = tf.withName(truePositivesName).variable(zeros.call(tf.constant(Shape.of(this.thresholds.length)), TFloat32.DTYPE));
            this.addVariable(truePositivesName, truePositives, zeros);
        }
        this.falseNegatives = getVariable(falseNegativesName);
        if (this.falseNegatives == null) {
            
            falseNegatives = tf.withName(falseNegativesName).variable(zeros.call(tf.constant(Shape.of(this.thresholds.length)), TFloat32.DTYPE));
            this.addVariable(falseNegativesName, falseNegatives, zeros);
        }
    }

    @Override
    public List<Op> updateStateList(Operand... args) {
         Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeights = args.length > 2 ? args[2] : null;
        
        Map<ConfusionMatrixEnum, Variable> confusionMatrix = new HashMap<>();
        confusionMatrix.put(ConfusionMatrixEnum.TRUE_POSITIVES, this.truePositives);
        confusionMatrix.put(ConfusionMatrixEnum.FALSE_NEGATIVES, this.falseNegatives);

        List<Op> updateOperations = new ArrayList<>();
        updateOperations.addAll(Metrics.update_confusion_matrix_variables(tf,
                confusionMatrix,
                Collections.EMPTY_MAP,
                yTrue,
                yPred, 
                this.thresholds,
                this.topK,
                this.classID,
                sampleWeights, false, null));
        return updateOperations;
    }

    @Override
    public Operand result(Ops rtf) {
        Operand result = rtf.math.divNoNan(this.truePositives,
                                 rtf.math.add(this.truePositives, this.falseNegatives));
        return this.thresholds.length == 1 ?  
                rtf.slice(result, rtf.constant(new int[] { 0 }), rtf.constant(new int[1]))
                : result;
    }
    
    public float[] getThresholds() {
        return this.thresholds;
    }
    public Integer getTopK() {
        return this.topK;
    }
    public Integer getClassID() {
        return this.classID;
    }
    
    public Variable<TFloat32> getTruePositives() {
        return this.truePositives;
    }
    public Variable<TFloat32> getFalseNegatives() {
        return this.falseNegatives;
    }
}
