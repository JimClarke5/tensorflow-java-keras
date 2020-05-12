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
public class Adam extends org.tensorflow.framework.optimizers.Adam implements OptimizerInterface  {
    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String EPSILON_KEY = "epsilon";
    public static final String BETA_ONE_KEY = "beta_1";
    public static final String BETA_TWO_KEY = "beta_2";
    
    public static final float  LEARNING_RATE_DEFAULT = 0.001F;
    public static final float  EPSILON_DEFAULT = 1e-07F;
    public static final float  BETA_ONE_DEFAULT = 0.9F;
    public static final float  BETA_TWO_DEFAULT = 0.999F;
    

    /**
     * create an Adam
     * @param graph
     */
    public Adam(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    public Adam(Graph graph, String name) {
        this(graph, name, LEARNING_RATE_DEFAULT, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    public Adam(Graph graph, float learningRate) {
        this(graph, learningRate, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    public Adam(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    public Adam(Graph graph, float learningRate, float betaOne, float betaTwo, float epsilon) {
        super(graph, learningRate, betaOne, betaTwo, epsilon);
        initConfig(learningRate, betaOne, betaTwo, epsilon);
    }
    public Adam(Graph graph, String name, float learningRate, float betaOne, float betaTwo, float epsilon) {
        super(graph, name, learningRate, betaOne, betaTwo, epsilon);
        initConfig(learningRate, betaOne, betaTwo, epsilon);
    }
    

    /**
     * create an Adam
     *
     * @param graph
     * @param config a config object to initialize
     */
    public static  Adam create(Graph graph, Map<String, Object> config) {
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float epsilon = (float)config.getOrDefault(EPSILON_KEY, EPSILON_DEFAULT);
        float betaOne = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float betaTwo = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        if(name == null) 
            return new Adam(graph, learningRate, betaOne,betaTwo, epsilon );
        else
            return new Adam(graph, name, learningRate, betaOne,betaTwo, epsilon );
    }
    
    protected void initConfig(float learningRate, float betaOne, float betaTwo, float epsilon) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(EPSILON_KEY, epsilon);
        config.put(BETA_ONE_KEY, betaOne);
        config.put(BETA_TWO_KEY, betaTwo);
        
    }
}
