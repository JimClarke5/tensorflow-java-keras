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
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase;
import static org.tensorflow.keras.metrics.impl.SensitivitySpecificityBase.DEFAULT_NUM_THRESHOLDS;
import org.tensorflow.op.Ops;

/**
 * Computes the maximally achievable recall at a required precision.
 *
 * @author jbclarke
 */
public class RecallAtPrecision extends SensitivitySpecificityBase {

    private float precision;

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param precision the precision. A scalar value in range [0, 1]
     */
    public RecallAtPrecision(Ops tf, float precision) {
        this(tf, null, precision, DEFAULT_NUM_THRESHOLDS, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'auc'
     * @param precision the precision. A scalar value in range [0, 1]
     */
    public RecallAtPrecision(Ops tf, String name, float precision) {
        this(tf, name, precision, DEFAULT_NUM_THRESHOLDS, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param precision the precision. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for
     * matching the given precision.
     */
    public RecallAtPrecision(Ops tf, float precision, int numThresholds) {
        this(tf, null, precision, numThresholds, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param precision the precision. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for
     * matching the given precision.
     */
    public RecallAtPrecision(Ops tf, String name, float precision, int numThresholds) {
        this(tf, name, precision, numThresholds, null);
    }

    /**
     * Creates a PrecisionRecall metric.
     *
     * @param tf The TensorFlow Ops
     * @param name the name of the metric, default is 'PrecisionAtRecall'
     * @param precision the precision. A scalar value in range [0, 1]
     * @param numThresholds Defaults to 200. The number of thresholds to use for
     * matching the given precision.
     * @param dType the type of the metric result.
     */
    public RecallAtPrecision(Ops tf, String name, float precision, int numThresholds, DataType dType) {
        super(tf, name, precision, numThresholds, dType);
        assert precision >= 0f && precision <= 1f :
                "`precision` must be in the range [0, 1].";
        this.precision = precision;
    }

    @Override
    public Operand result(Ops rtf) {

        Operand precisions = rtf.math.divNoNan(
                this.truePositives, rtf.math.add(this.truePositives, this.falsePositives));
        Operand recalls = rtf.math.divNoNan(
                this.truePositives, rtf.math.add(this.truePositives, this.falseNegatives));

        Operand isFeasible = rtf.math.greaterEqual(precisions, rtf.constant(this.value));
        Operand feasible = rtf.where(isFeasible);
        Operand feasibleExists = rtf.math.greater(rtf.size(feasible), rtf.constant(0));

        Operand gather = rtf.expandDims(rtf.gather(recalls, feasible, rtf.constant(0)), rtf.constant(0));
        Operand bestRecall = rtf.select(feasibleExists,
                rtf.reduceMax(gather, K.allAxis(rtf, gather)), rtf.constant(0.0f));
        return bestRecall;

    }

    /**
     * @return the precision
     */
    public float getPrecision() {
        return precision;
    }
}
