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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;

/**
 * Test cases for GlorotNormal initializer
 */
public class GlorotNormalTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    private static final long SEED = 1000L;

    int counter;

    public GlorotNormalTest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of getConfig method, of class GlorotNormal.
     */
    @Test
    public void testGetConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotNormal.SCALE_KEY, 1.0);
        config.put(GlorotNormal.MODE_KEY, "fan_avg");
        config.put(GlorotNormal.DISTRIBUTION_KEY, "truncated_normal");
        config.put(GlorotNormal.SEED_KEY, SEED);
        GlorotNormal instance = new GlorotNormal(null, SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of getConfig method, of class GlorotNormal.
     */
    @Test
    public void testConfigCTORMap() {
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotNormal.SCALE_KEY, 1.0);
        config.put(GlorotNormal.MODE_KEY, "fan_avg");
        config.put(GlorotNormal.DISTRIBUTION_KEY, "truncated_normal");
        config.put(GlorotNormal.SEED_KEY, SEED);
        GlorotNormal instance = new GlorotNormal(null, config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class GlorotNormal.
     */
    @Test
    public void testCall_Float() {
        float[] expected = {-0.52388954F, -0.29329166F, -0.07872587F, -0.31851602F};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            GlorotNormal<TFloat32> instance
                    = new GlorotNormal<>(tf, SEED);
            Operand<TFloat32> operand = instance.call(tf.constant(shape), TFloat32.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    @Test
    public void testCall_Double() {
        double[] expected = {1.4971264721246893, -1.2488522307109322,
            -0.5409677352523339, 0.4871390504288623};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            GlorotNormal<TFloat64> instance
                    = new GlorotNormal<>(tf, SEED);
            Operand<TFloat64> operand = instance.call(tf.constant(shape), TFloat64.DTYPE);
            session.evaluate(expected, operand);
        }
    }

}
