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
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.PrintUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class RandomNormalTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    private static final long SEED = 1000L;
    private static final double MEAN_VALUE = 0.0;
    private static final double STDDEV_VALUE = 3.0;
    
    
    public RandomNormalTest() {
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
     * Test of getConfig method, of class RandomNormal.
     */
    @Test
    public void testGetConfig() {
        System.out.println("getConfig");
        Map<String, Object> config = new HashMap<>();
        config.put(RandomNormal.MEAN_KEY, MEAN_VALUE);
        config.put(RandomNormal.STDDEV_KEY, STDDEV_VALUE);
        config.put(RandomNormal.SEED_KEY, SEED);    
        RandomNormal instance = new RandomNormal(null, MEAN_VALUE, STDDEV_VALUE, SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    
    /**
     * Test of getConfig method, of class RandomNormal.
     */
    @Test
    public void testConfigCTORMap() {
        System.out.println("ctor Map");
        Map<String, Object> config = new HashMap<>();
        config.put(RandomNormal.MEAN_KEY, MEAN_VALUE);
        config.put(RandomNormal.STDDEV_KEY, STDDEV_VALUE);
        config.put(RandomNormal.SEED_KEY, SEED);    
        RandomNormal instance = new RandomNormal(null, config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

   
    
    
    /**
     * Test of call method, of class RandomNormal.
     */
    @Test
    public void testCall_Float() {
        System.out.println("call Float");
        float[] actual = { 0,0, 0, 0};
        float[] expected = {  -1.955122F, -1.0945456F, -0.29379985F, -1.1886811F };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            RandomNormal<TFloat32> instance = 
                    new RandomNormal(tf, MEAN_VALUE, STDDEV_VALUE, SEED);
            Operand<TFloat32> operand = instance.call(tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat32(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    @Test
    public void testCall_Double() {
        System.out.println("call Double");
        double[] actual = { 0,0, 0, 0};
        double[] expected = { 5.58717960737721, -4.6606361225803825, 
                            -0.5743065932046001, -7.207274031929497  };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            RandomNormal<TFloat64> instance = 
                    new RandomNormal(tf, MEAN_VALUE, STDDEV_VALUE, SEED);
            Operand<TFloat64> operand = instance.call(tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           PrintUtils.printTFloat64(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
}
