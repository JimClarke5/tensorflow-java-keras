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
import org.tensorflow.keras.metrics.impl.ConfusionMatrixConditionCount;
import org.tensorflow.keras.metrics.impl.ConfusionMatrixEnum;
import org.tensorflow.op.Ops;

/**
 * Calculates how often predictions equals labels.
 * @author Jim Clarke
 */
public class FalsePositives extends ConfusionMatrixConditionCount {
    
    public static final String DEFAULT_NAME = "false_positived";
    private static final float[] DEFAULT_THRESHOLD = new float[] {0.5f};
    
    
    public FalsePositives(Ops tf) {
        this(tf, DEFAULT_NAME, DEFAULT_THRESHOLD, null);
    }
    
    public FalsePositives(Ops tf, float threshold) {
        this(tf, DEFAULT_NAME, new float[]{ threshold }, null);
    }
    public FalsePositives(Ops tf, float[] thresholds) {
        this(tf, DEFAULT_NAME, thresholds, null);
    }
    
    public FalsePositives(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, DEFAULT_THRESHOLD, dType);
    }
    public FalsePositives(Ops tf, String name) {
        this(tf, name, DEFAULT_THRESHOLD, null);
    }
    
    public FalsePositives(Ops tf, String name, float threshold) {
        this(tf, name, new float[]{ threshold }, null);
    }
    
    public FalsePositives(Ops tf, String name, float[] thresholds) {
        this(tf, name, thresholds , null);
    }
    
    public FalsePositives(Ops tf, String name, DataType dType) {
        this(tf, name, DEFAULT_THRESHOLD, dType);
    }
    
    public FalsePositives(Ops tf, String name,  float threshold, DataType dType) {
        this(tf, name, new float[]{ threshold}, dType);
    }
    
    public FalsePositives(Ops tf, String name, float[] thresholds, DataType dType) {
        super(tf, name,  ConfusionMatrixEnum.FALSE_POSITIVES, thresholds, dType);
    }


    
}
