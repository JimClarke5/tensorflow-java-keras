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
 * Computes the precision at a given specificity.
 * 
 * @author jbclarke
 */
public class SensitivityAtSpecificity extends SensitivitySpecificityBase {
    
    private final float specificity;
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param specificity  the specificity. A scalar value in range [0, 1]
     */
    public SensitivityAtSpecificity(Ops tf, float specificity) {
        this(tf, null, specificity, DEFAULT_NUM_THRESHOLDS, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param specificity the specificity. A scalar value in range [0, 1]
     */
    public SensitivityAtSpecificity(Ops tf, String name, float specificity) {
        this(tf, name, specificity, DEFAULT_NUM_THRESHOLDS, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param specificity the specificity. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given specificity.
     */
    public SensitivityAtSpecificity(Ops tf, float specificity, int numThresholds) {
        this(tf, null, specificity, numThresholds, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param specificity the specificity. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given specificity.
     */
    public SensitivityAtSpecificity(Ops tf, String name, float specificity, int numThresholds) {
        this(tf, name, specificity, numThresholds, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param specificity the specificity. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given specificity.
     * @param dType the type of the metric result.
     */
    public SensitivityAtSpecificity(Ops tf, String name, float specificity, int numThresholds, DataType dType) {
        super(tf, name, specificity, numThresholds, dType);
        assert specificity >= 0f && specificity <= 1f :
                "`specificity` must be in the range [0, 1].";
        this.specificity = specificity;
    }
    
    


    @Override
    public Operand result(Ops rtf) {
        
        Operand specificitys = rtf.math.divNoNan(
                this.trueNegatives, rtf.math.add(this.trueNegatives, this.falsePositives));
        Operand sub = rtf.math.sub(specificitys, rtf.constant(this.getValue()));
        Operand minIndex = rtf.math.argMin(
                rtf.math.abs(sub), rtf.constant(0), TInt32.DTYPE );
        minIndex = rtf.expandDims(minIndex, rtf.constant(0));
        
        Operand trueSlice = rtf.slice(this.truePositives, minIndex, rtf.constant(new int[]{1}));
        Operand falseSlice = rtf.slice(this.falseNegatives, minIndex, rtf.constant(new int[]{1}));
        Operand result =  rtf.math.divNoNan( trueSlice, rtf.math.add(trueSlice, falseSlice));
        return result;
        
        
    }

    /**
     * @return the specificity
     */
    public float getSpecificity() {
        return specificity;
    }


}
