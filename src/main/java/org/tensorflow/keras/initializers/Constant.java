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
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 * Initializer that generates tensors with a constant value.
 * @author Jim Clarke
 * @param <U> The Type for the call operation
 */
public class Constant<U extends TType> extends Initializer<U> {
    public final String VALUE_KEY = "value";
    public final String BOOL_VALUE_KEY = "bvalue";

    private final Double value;
    private final Boolean bvalue;
    
    /** 
     * creates an Initializer that generates tensors with a constant value.
     * @param value a number value
     */
    public Constant(double value) {
        super();
        this.value = value;
        bvalue = null;
    }
    
    /**
     * creates an Initializer that generates tensors with a constant value.
     * @param bvalue a boolean value
     */
     public Constant(boolean bvalue) {
        super();
        this.bvalue = bvalue;
        this.value = null;
    }
    
    
    /**
     *  Creates an Initializer that generates tensors with a constant value.
     * @param config the config object used to initialize this Matrix
     */
    public Constant(Map<String, Object> config) {
        super(config);
        this.value = (Double)config.get(VALUE_KEY);
        this.bvalue = (Boolean)config.get(BOOL_VALUE_KEY);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getConfig() {
        Map<String, Object> config = super.getConfig();
        config.put(VALUE_KEY, value);
        config.put(BOOL_VALUE_KEY, bvalue);
        return config;
    }
    
    

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Ops tf, Operand<TInt64> dims, DataType<U> dtype) {
        assert(TypeUtils.isNumeric(dtype) || TypeUtils.isBoolean(dtype)) : 
                "DataType must be numeric or boolean: " + dtype.name();
        if(this.value != null) {
            return tf.fill(dims, tf.dtypes.cast(tf.constant(value), dtype));
        }else {
            return tf.fill(dims, tf.dtypes.cast(tf.constant(this.bvalue), dtype));
        }
    }
    
}
