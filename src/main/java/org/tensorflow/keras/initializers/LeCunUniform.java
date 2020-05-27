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
package org.tensorflow.keras.initializers;

import java.util.Map;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * LeCun uniform initializer.
 * 
 * @author Jim Clarke
 */
public class LeCunUniform<U extends TType> extends VarianceScaling<U> {

    /**
     * Creates the LeCun uniform initializer.
     * 
     * @param tf the TensorFlow Ops
     */
    public LeCunUniform(Ops tf) {
        super(tf, 1.0, "fan_in", "uniform", null);
    }
    
    /**
     * Creates the LeCun uniform initializer.
     * 
     * @param tf the TensorFlow Ops
     * @param seed the seed for random number generation
     */
    public LeCunUniform(Ops tf, Long seed) {
        super(tf, 1.0, "fan_in", "uniform", seed);
    }
    
    /**
     * Creates the LeCun uniform initializer.
     * 
     * @param tf the TensorFlow Ops
     * @param config the config object used to initialize the Initializer values
     */
    public LeCunUniform(Ops tf, Map<String, Object> config) {
        super(tf, config);
    }
    
}
