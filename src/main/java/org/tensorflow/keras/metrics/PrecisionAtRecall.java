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
 * Computes the precision at a given recall.
 * 
 * @author jbclarke
 */
public class PrecisionAtRecall extends SensitivitySpecificityBase {
    
    private final float recall;
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param recall  the recall. A scalar value in range [0, 1]
     */
    public PrecisionAtRecall(Ops tf, float recall) {
        this(tf, null, recall, DEFAULT_NUM_THRESHOLDS, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param recall the recall. A scalar value in range [0, 1]
     */
    public PrecisionAtRecall(Ops tf, String name, float recall) {
        this(tf, name, recall, DEFAULT_NUM_THRESHOLDS, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param recall the recall. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given recall.
     */
    public PrecisionAtRecall(Ops tf, float recall, int numThresholds) {
        this(tf, null, recall, numThresholds, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param recall the recall. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given recall.
     */
    public PrecisionAtRecall(Ops tf, String name, float recall, int numThresholds) {
        this(tf, name, recall, numThresholds, null);
    }
    
    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param recall the recall. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for matching the given recall.
     * @param dType the type of the metric result.
     */
    public PrecisionAtRecall(Ops tf, String name, float recall, int numThresholds, DataType dType) {
        super(tf, name, recall, numThresholds, dType);
        assert recall >= 0f && recall <= 1f :
                "`recall` must be in the range [0, 1].";
        this.recall = recall;
    }
    
    


    @Override
    public Operand result(Ops rtf) {
        
        Operand recalls = rtf.math.divNoNan(
                this.truePositives, rtf.math.add(this.truePositives, this.falseNegatives));
        Operand sub = rtf.math.sub(recalls, rtf.constant(this.getValue()));
        Operand minIndex = rtf.math.argMin(
                rtf.math.abs(sub), rtf.constant(0), TInt32.DTYPE );
        minIndex = rtf.expandDims(minIndex, rtf.constant(0));
        
        Operand trueSlice = rtf.slice(this.truePositives, minIndex, rtf.constant(new int[]{1}));
        Operand falseSlice = rtf.slice(this.falsePositives, minIndex, rtf.constant(new int[]{1}));
        Operand result =  rtf.math.divNoNan( trueSlice, rtf.math.add(trueSlice, falseSlice));
        return result;
        
        
    }

    /**
     * @return the recall
     */
    public float getRecall() {
        return recall;
    }
}
