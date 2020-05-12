/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the );
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an  BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.optimizers;

import java.util.Map;
import org.tensorflow.Graph;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.config;

/**
 * Adam Optimizer that implements the Adam algorithm.
 *
 * @author Jim Clarke
 */
public class GradientDescent extends org.tensorflow.framework.optimizers.GradientDescent implements OptimizerInterface  {
    public static final String LEARNING_RATE_KEY = "learning_rate";
    
     public static final float  LEARNING_RATE_DEFAULT = 0.001F;
      /**
     * create an GradientDescent
     * @param graph
     */
    public GradientDescent(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT);
    }
    
    public GradientDescent(Graph graph, String name) {
        this(graph, name, LEARNING_RATE_DEFAULT);
    }
     public GradientDescent(Graph graph, float learningRate) {
        super(graph, learningRate);
        initConfig(learningRate);
    }
    
    public GradientDescent(Graph graph, String name, float learningRate) {
        super(graph, name, learningRate);
        initConfig(learningRate);
    }

    /**
     * create an Adam
     *
     * @param graph
     * @param config a config object to initialize
     * @return 
     */
    public static  GradientDescent create(Graph graph, Map<String, Object> config) {
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        if(name == null)
            return new GradientDescent(graph, learningRate);
        else
            return new GradientDescent(graph, name, learningRate);
    }
    
    private void initConfig(float learningRate) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
    }
}
