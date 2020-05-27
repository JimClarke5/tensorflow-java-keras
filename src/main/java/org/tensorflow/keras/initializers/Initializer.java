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

import java.util.HashMap;
import java.util.Map;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * Abstract class for all Initializers
 * 
 * @author Jim Clarke
 * @param <U> The Type for the call operation
 */
public abstract class Initializer<U extends TType> implements InitializerFunction<U> {
    
        
    
    private final Map<String, Object> config;
    
    protected final Ops tf;
    
    /** 
     * Create an Initializer
     * @param tf the TensorFlow Ops
     */
    protected Initializer(Ops tf) {
        this(tf, new HashMap<>());
    }
    
    // TODO do we need this.
    /**
     *  Create an Initializer
     * 
     * @param tf the TensorFlow Ops
     * @param config the config mao opbject
     */
    protected Initializer(Ops tf, Map<String, Object> config) {
        this.config = config;
        this.tf = tf;
    }
    
    
    /**
     * @return the config object used to initialize the Initializer values
     */
    public Map<String, Object> getConfig() {
        return config;
    }

    /**
     * @return the tf
     */
    public Ops getTf() {
        return tf;
    }


    
}
