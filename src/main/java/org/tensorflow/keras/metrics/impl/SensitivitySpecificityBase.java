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
package org.tensorflow.keras.metrics.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.tf.ControlDependencies;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;

/**
 * Abstract base class for computing sensitivity and specificity.
 * @author jbclarke
 */
public abstract class SensitivitySpecificityBase extends Metric {
    public static final int DEFAULT_NUM_THRESHOLDS = 200;
    
    public static final String TRUE_POSITIVES = "TRUE_POSITIVES";
    public static final String FALSE_POSITIVES = "FALSE_POSITIVES";
    public static final String TRUE_NEGATIVES = "TRUE_NEGATIVES";
    public static final String FALSE_NEGATIVES = "FALSE_NEGATIVES";
    
    protected Variable<TFloat32> truePositives;
    protected Variable<TFloat32> falsePositives;
    protected Variable<TFloat32> trueNegatives;
    protected Variable<TFloat32> falseNegatives;
    
    private final String truePositivesName;
    private final String falsePositivesName;
    private final String trueNegativesName;
    private final String falseNegativesName;
    
    protected final int numThresholds;
    protected final float value;
    protected final float[] thresholds;
    
    /**
     * Creates a SensitivitySpecificityBase 
     * @param tf the TensorFlow Ops
     * @param name the name of the metric instance 
     * @param value A scalar value in range `[0, 1]`
     * @param numThresholds The number of thresholds to
     *  use for matching the given recall.
     * @param dType the data type of the metric result
     */
    
    protected SensitivitySpecificityBase(Ops tf, String name, float value, int numThresholds, DataType dType) {
        super(tf, name, dType);
        assert numThresholds > 0 : "`num_thresholds` must be > 0.";
        this.truePositivesName = this.getVariableName(TRUE_POSITIVES);
        this.falsePositivesName = this.getVariableName(FALSE_POSITIVES);
        this.trueNegativesName = this.getVariableName(TRUE_NEGATIVES);
        this.falseNegativesName = this.getVariableName(FALSE_NEGATIVES);
        
        this.value = value;
        this.numThresholds = numThresholds;
        
        if(this.numThresholds == 1) {
            this.thresholds = new float[]{0.5f};
        }else {
            this.thresholds = new float[numThresholds];
            for(int i = 0; i < numThresholds-2; i++) {
                this.thresholds[i+1] = (float)(i + 1f) / (float)(numThresholds - 1);
            }
            this.thresholds[numThresholds-1] = 1f;
        }
        init();
    }
    
    /**
     * Initialize the Variables
     */
    private void init() {
        Zeros zeros = new Zeros(tf);
        
        this.truePositives = getVariable(truePositivesName);
        if (this.getTruePositives() == null) {
            
            truePositives = tf.withName(truePositivesName).variable(zeros.call(tf.constant(Shape.of(getNumThresholds())), TFloat32.DTYPE));
            this.addVariable(truePositivesName, getTruePositives(), zeros);
        }
        this.falsePositives = getVariable(falsePositivesName);
        if (this.getFalsePositives() == null) {
            
            falsePositives = tf.withName(falsePositivesName).variable(zeros.call(tf.constant(Shape.of(getNumThresholds())), TFloat32.DTYPE));
            this.addVariable(falsePositivesName, getFalsePositives(), zeros);
        }
        this.trueNegatives = getVariable(trueNegativesName);
        if (this.getTrueNegatives() == null) {
            
            trueNegatives = tf.withName(trueNegativesName).variable(zeros.call(tf.constant(Shape.of(getNumThresholds())), TFloat32.DTYPE));
            this.addVariable(trueNegativesName, getTrueNegatives(), zeros);
        }
        this.falseNegatives = getVariable(falseNegativesName);
        if (this.getFalseNegatives() == null) {
            
            falseNegatives = tf.withName(falseNegativesName).variable(zeros.call(tf.constant(Shape.of(getNumThresholds())), TFloat32.DTYPE));
            this.addVariable(falseNegativesName, getFalseNegatives(), zeros);
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public List<Op> updateStateList(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeights = args.length > 2 ? args[2] : null;
        
        Map<ConfusionMatrixEnum, Variable> confusionMatrix = new HashMap<>();
        confusionMatrix.put(ConfusionMatrixEnum.TRUE_POSITIVES, this.getTruePositives());
        confusionMatrix.put(ConfusionMatrixEnum.FALSE_POSITIVES, this.getFalsePositives());
        confusionMatrix.put(ConfusionMatrixEnum.TRUE_NEGATIVES, this.getTrueNegatives());
        confusionMatrix.put(ConfusionMatrixEnum.FALSE_NEGATIVES, this.getFalseNegatives());

        List<Op> updateOperations = new ArrayList<>();
        updateOperations.addAll(Metrics.update_confusion_matrix_variables(tf,
                confusionMatrix,
                Collections.EMPTY_MAP,
                yTrue,
                yPred, this.getThresholds(),
                null,
                null,
                sampleWeights, false, null));
        return updateOperations;
    }

    /**
     * @return the truePositives
     */
    public Variable<TFloat32> getTruePositives() {
        return truePositives;
    }

    /**
     * @return the falsePositives
     */
    public Variable<TFloat32> getFalsePositives() {
        return falsePositives;
    }

    /**
     * @return the trueNegatives
     */
    public Variable<TFloat32> getTrueNegatives() {
        return trueNegatives;
    }

    /**
     * @return the falseNegatives
     */
    public Variable<TFloat32> getFalseNegatives() {
        return falseNegatives;
    }

    /**
     * @return the numThresholds
     */
    public int getNumThresholds() {
        return numThresholds;
    }

    /**
     * @return the value
     */
    public float getValue() {
        return value;
    }

    /**
     * @return the thresholds
     */
    public float[] getThresholds() {
        return thresholds;
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

    /**
     * @return the trueNegativesName
     */
    public String getTrueNegativesName() {
        return trueNegativesName;
    }

    /**
     * @return the falseNegativesName
     */
    public String getFalseNegativesName() {
        return falseNegativesName;
    }
    
}
