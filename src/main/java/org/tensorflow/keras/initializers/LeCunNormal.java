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
import org.tensorflow.types.family.TType;

/**
 * LeCun normal initializer.
 * @author Jim Clarke
 * @param <U> The Type for the call operation
 */
public class LeCunNormal<U extends TType> extends VarianceScaling<U> {

    /**
     * create an LeCunNormal Initializer
     */
    public LeCunNormal() {
        super(1.0, "fan_in", "truncated_normal", null);
    }
    
    /**
     * create an LeCunNormal Initializer
     * @param seed the seed for random number generation
     */
    public LeCunNormal(Long seed) {
        super(1.0, "fan_in", "truncated_normal", seed);
    }
    
    /**
     * create a LeCunNormal initializer
     * @param config the config object used to initialize the Initializer values
     */
    public LeCunNormal(Map<String, Object> config) {
        super(config);
    }
    
}
