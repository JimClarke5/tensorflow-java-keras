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
package org.tensorflow.keras.constraints;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.metrics.impl.MetricsImpl;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class MaxNormTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public MaxNormTest() {
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
     * Test of getConfig method, of class Constant.
     */
    @Test
    public void testConfig() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MaxNorm instance = new MaxNorm(tf, 1.0f, 1);
            
            assertEquals(1.0f, instance.getMaxValue());
            assertEquals(1, instance.getAxis());
        }
    }
    
    
    private float[] getSampleArray() {
        Random rand = new Random(3537l);
        float[] result = new float[100 * 100];
        for(int i = 0; i < result.length; i++) {
            result[i] = rand.nextFloat() * 100 - 50;
        }
        result[0] = 0;
        return result;
    }
    
    /**
     * Test of call method, of class MaxNorm.
     */
    @Test
    public void testCall() {
        float[] testValues = {0.1f, 0.5f, 3f, 8f, 1e-7f};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            final float[] array = getSampleArray();
            Operand<TFloat32> weights = tf.reshape(tf.constant(array), tf.constant(Shape.of(100,100)));
            for(AtomicInteger i = new AtomicInteger(); i.get() < testValues.length; i.getAndIncrement() ) {
                MaxNorm instance = new MaxNorm(tf, testValues[i.get()]);
                Operand result = instance.call(weights);
                session.evaluate(result, (Number v) ->  v.floatValue() <= testValues[i.get()]);
            }
        }
    }
    /**
     * Test of call method, of class MaxNorm.
     */
    @Test
    public void testCall1() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            MaxNorm instance = new MaxNorm(tf, 2.0f);
            Operand<TFloat32> weights = tf.constant(new float[][] {{0,1, 3, 3}, {0, 0, 0, 3}, {0, 0, 0, 3},});
            MetricsImpl.setDebug(session.getGraphSession());
            Operand result = instance.call(weights);
            float[] expected =  {
                0, 1, 2, 1.1547005f,
                0, 0, 0, 1.1547005f,
                0, 0, 0, 1.1547005f
            };
            session.evaluate(expected, result);
            
            MetricsImpl.setDebug(null);
        }
        
    }
    
}
