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
import org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt32;

/**
 * Computes the precision at a given sensitivity.
 * 
 * @author jbclarke
 */
public class SpecificityAtSensitivity extends SensitivitySpecificityBase {
    
    private final float sensitivity;
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param sensitivity  the sensitivity. A scalar value in range [0, 1]
     */
    public SpecificityAtSensitivity(Ops tf, float sensitivity) {
        this(tf, null, sensitivity, DEFAULT_NUM_THRESHOLDS, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param sensitivity the sensitivity. A scalar value in range [0, 1]
     */
    public SpecificityAtSensitivity(Ops tf, String name, float sensitivity) {
        this(tf, name, sensitivity, DEFAULT_NUM_THRESHOLDS, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param sensitivity the sensitivity. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given sensitivity.
     */
    public SpecificityAtSensitivity(Ops tf, float sensitivity, int numThresholds) {
        this(tf, null, sensitivity, numThresholds, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param sensitivity the sensitivity. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given sensitivity.
     */
    public SpecificityAtSensitivity(Ops tf, String name, float sensitivity, int numThresholds) {
        this(tf, name, sensitivity, numThresholds, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param sensitivity the sensitivity. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given sensitivity.
     * @param dType the type of the metric result.
     */
    public SpecificityAtSensitivity(Ops tf, String name, float sensitivity, int numThresholds, DataType dType) {
        super(tf, name, sensitivity, numThresholds, dType);
        assert sensitivity >= 0f && sensitivity <= 1f :
                "`sensitivity` must be in the range [0, 1].";
        this.sensitivity = sensitivity;
    }
    
    


    @Override
    public Operand result() {
        
        Operand sensitivitys = tf.math.divNoNan(
                this.truePositives, tf.math.add(this.truePositives, this.falseNegatives));
        Operand sub = tf.math.sub(sensitivitys, tf.constant(this.getValue()));
        Operand minIndex = tf.math.argMin(
                tf.math.abs(sub), tf.constant(0), TInt32.DTYPE );
        minIndex = tf.expandDims(minIndex, tf.constant(0));
        
        Operand trueSlice = tf.slice(this.trueNegatives, minIndex, tf.constant(new int[]{1}));
        Operand falseSlice = tf.slice(this.falsePositives, minIndex, tf.constant(new int[]{1}));
        Operand result =  tf.math.divNoNan( trueSlice, tf.math.add(trueSlice, falseSlice));
        return result;
        
        
    }

    /**
     * @return the sensitivity
     */
    public float getSensitivity() {
        return sensitivity;
    }


}
