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
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;

/**
 * Computes the precision of the predictions with respect to the labels.
 * @author jbclarke
 */
public class Precision extends Metric {
    public static final String TRUE_POSITIVES = "TRUE_POSITIVES";
    public static final String FALSE_POSITIVES = "FALSE_POSITIVES";
    
    private final float[] thresholds;
    private final Integer topK;
    private final Integer classId;
    
    private Variable<TFloat32> truePositives;
    private Variable<TFloat32> falsePositives;
    private final String truePositivesName;
    private final String falsePositivesName;
    
    
    
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     */
    public  Precision(Ops tf) {
        this(tf, null, new float[] {0.5f}, null, null, null);
    }
    
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     */
    public Precision(Ops tf, String name) {
        this(tf, name, new float[] {0.5f}, null, null, null);
    }
    
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param threshold A float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     */
    public  Precision(Ops tf, float threshold) {
        this(tf, null,new float[]{threshold}, null, null, null);
    }
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param thresholds Optional float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     */
    public  Precision(Ops tf, float[] thresholds) {
        this(tf, null,thresholds, null, null, null);
    }
    
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     * @param threshold A float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     */
    public  Precision(Ops tf, String name, float threshold) {
        this(tf, name, new float[]{threshold}, null, null, null);
    }
    
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     * @param thresholds Optional float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     */
    public  Precision(Ops tf, String name, float[] thresholds) {
        this(tf, name,thresholds, null, null, null);
    }
    
     /**
     * 
     * @param tf the TensorFlow Ops
     * @param threshold  A float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     * @param topK An optional value specifying the top-k predictions 
     * to consider when calculating precision.
     * @param classId Optional Integer class ID for which we want binary metrics.
     * This must be in the half-open interval [0, num_classes), 
     * where num_classes is the last dimension of predictions.
     */
    public  Precision(Ops tf, float threshold,  Integer topK, Integer classId) {
        this(tf, null,new float[]{threshold}, topK, classId, null);
    }
    
    /**
     * 
     * @param tf the TensorFlow Ops
     * @param thresholds  Optional float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     * @param topK An optional value specifying the top-k predictions 
     * to consider when calculating precision.
     * @param classId Optional Integer class ID for which we want binary metrics.
     * This must be in the half-open interval [0, num_classes), 
     * where num_classes is the last dimension of predictions.
     */
    public  Precision(Ops tf, float[] thresholds,  Integer topK, Integer classId) {
        this(tf, null,thresholds, topK, classId, null);
    }
    
    /**
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     * @param threshold  A float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     * @param topK An optional value specifying the top-k predictions 
     * to consider when calculating precision.
     * @param classId Optional Integer class ID for which we want binary metrics.
     * This must be in the half-open interval [0, num_classes), 
     * where num_classes is the last dimension of predictions.
     */
    public  Precision(Ops tf, String name, float threshold, 
            Integer topK, Integer classId) {
        this(tf, name,new float[]{threshold}, topK, classId, null);
    }
    
    /**
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     * @param thresholds  Optional float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     * @param topK An optional value specifying the top-k predictions 
     * to consider when calculating precision.
     * @param classId Optional Integer class ID for which we want binary metrics.
     * This must be in the half-open interval [0, num_classes), 
     * where num_classes is the last dimension of predictions.
     */
    public  Precision(Ops tf, String name, float[] thresholds, 
            Integer topK, Integer classId) {
        this(tf, name,thresholds, topK, classId, null);
    }
    
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     * @param threshold  A float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     * @param topK An optional value specifying the top-k predictions 
     * to consider when calculating precision.
     * @param classId Optional Integer class ID for which we want binary metrics.
     * This must be in the half-open interval [0, num_classes), 
     * where num_classes is the last dimension of predictions.
     * @param dType  data type of the metric result
     */
    public <U extends TNumber> Precision(Ops tf, String name,
            float threshold, Integer topK, Integer classId,  DataType<U> dType) {
        this(tf, name,new float[]{threshold}, topK, classId, dType);
    }
    /**
     * Creates a Precision Metric.
     * 
     * @param tf the TensorFlow Ops
     * @param name name of the metric instance
     * @param thresholds  Optional float threshold values in the range [0, 1]. 
     * A threshold is compared with prediction values to determine
     * the truth value of predictions.
     * @param topK An optional value specifying the top-k predictions 
     * to consider when calculating precision.
     * @param classId Optional Integer class ID for which we want binary metrics.
     * This must be in the half-open interval [0, num_classes), 
     * where num_classes is the last dimension of predictions.
     * @param dType  data type of the metric result
     */
    public <U extends TNumber> Precision(Ops tf, String name,
            float[] thresholds, Integer topK, Integer classId,  DataType<U> dType) {
        super(tf, name, dType);
        this.truePositivesName = this.getVariableName(TRUE_POSITIVES);
        this.falsePositivesName = this.getVariableName(FALSE_POSITIVES);
        float defaultThreshold = topK == null ? 0.5f : Metrics.NEG_INF;
        this.thresholds =  thresholds == null ? new float[]{defaultThreshold} : thresholds ;
        this.topK = topK;
        this.classId = classId;
        
        init();
     }
    
    private void init() {
        Zeros zeros = new Zeros(tf);
        
        this.truePositives =  getVariable(truePositivesName);
        if (this.truePositives == null) {
            this.truePositives =  tf.withName(truePositivesName).variable(zeros.call(tf.constant(Shape.of(this.getThresholds().length)), TFloat32.DTYPE));
            this.addVariable(truePositivesName, this.truePositives, zeros);
            
        }
        this.falsePositives =  getVariable(falsePositivesName);
        if (this.falsePositives == null) {
            this.falsePositives =  tf.withName(falsePositivesName).variable(zeros.call(tf.constant(Shape.of(this.getThresholds().length)), TFloat32.DTYPE));
            this.addVariable(falsePositivesName, this.falsePositives, zeros);
        }
    }
    
    

    @Override
    public List<Op> updateStateList(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeight = args.length > 2? args[2]: null;
        
        Map<ConfusionMatrixEnum, Variable> confusionMatrix = new HashMap<>();
        confusionMatrix.put(ConfusionMatrixEnum.TRUE_POSITIVES, this.truePositives);
        confusionMatrix.put(ConfusionMatrixEnum.FALSE_POSITIVES, this.falsePositives);

        List<Op> updateOperations = new ArrayList<>();
        updateOperations.addAll(Metrics.update_confusion_matrix_variables(tf,
                confusionMatrix,
                Collections.EMPTY_MAP,
                yTrue,
                yPred, this.getThresholds(), this.getTopK(), this.getClassId(),
                sampleWeight, false, null));
        return updateOperations;
    }

    @Override
    public Operand result(Ops rtf) {
        
        Operand result = rtf.math.divNoNan(
                this.getTruePositives(), rtf.math.add(this.truePositives, this.falsePositives));
        return this.getThresholds().length == 1 ?
                rtf.slice(result, rtf.expandDims(rtf.constant(0), rtf.constant(0)), 
                        rtf.expandDims(rtf.constant(1), rtf.constant(0))) :
                result;
        
    }

    /**
     * @return the thresholds
     */
    public float[] getThresholds() {
        return thresholds;
    }

    /**
     * @return the topK
     */
    public Integer getTopK() {
        return topK;
    }

    /**
     * @return the classId
     */
    public Integer getClassId() {
        return classId;
    }

    /**
     * @return the truePositives
     */
    public Variable<TFloat32> getTruePositives() {
        return truePositives;
    }

    /**
     * @param truePositives the truePositives to set
     */
    public void setTruePositives(Variable<TFloat32> truePositives) {
        this.truePositives = truePositives;
    }

    /**
     * @return the falsePositives
     */
    public Variable<TFloat32> getFalsePositives() {
        return falsePositives;
    }

    /**
     * @param falsePositives the falsePositives to set
     */
    public void setFalsePositives(Variable<TFloat32> falsePositives) {
        this.falsePositives = falsePositives;
    }

    /**
     * @return the truePositivesName
     */
    public String getTruePositivesName() {
        return truePositivesName;
    }

    /**
     * @return the falsePositivesName
     */
    public String getFalsePositivesName() {
        return falsePositivesName;
    }
    
}
