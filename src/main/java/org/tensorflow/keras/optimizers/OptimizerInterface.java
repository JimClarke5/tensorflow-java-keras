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
package org.tensorflow.keras.optimizers;

import java.util.Map;
import org.tensorflow.Graph;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public interface OptimizerInterface {

    public static final String NAME_KEY = "name";
    
    public static  Graph assertGraph(Ops tf) {
        assert tf.scope().env().isGraph() :
                "Optimizers can only be used in Graph Mode";
        return (Graph)tf.scope().env();
    }

    /**
     * @return the config object used to initialize the Optimizer
     */
    public Map<String, Object> getConfig();

    /**
     * return the current learning rate
     *
     * @return the current learning rate
     */
    public float getLearningRate();

    /**
     * set the learning rate
     *
     * @param learningRate the learning rate;
     */
    public void setLearningRate(float learningRate);

}
