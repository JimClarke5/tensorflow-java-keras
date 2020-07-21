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
package org.tensorflow.keras.regularizers;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;

/**
 *
 * @author jbclarke
 */
public class L1Test extends CommonTest {
    
    
    public L1Test() {
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
     * Test of create method, of class AdaDelta.
     */
    @Test
    public void testCreate() {
         try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1 instance = new L1(tf, 0.2f);
            assertEquals(0.2f, instance.getL1());
            assertNull(instance.getL2());
            
            instance = new L1(tf, 0f);
            assertEquals(0.f, instance.getL1());
            assertNull(instance.getL2());
            
            instance = new L1(tf);
            assertEquals(Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL1());
            assertNull(instance.getL2());
            
        }
    }

   
    
    
    /**
     * Test of call method, of class L1L2.
     */
    @Test
    public void testCallNO() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1 instance = new L1(tf, 0.0f);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
            session.evaluate(0, result);
        }
    }
    
    
    /**
     * Test of call method, of class L1L2.
     */
    @Test
    public void testCallL1() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1 instance = new L1(tf);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
             float expected = regularizeL1(w, Regularizer.DEFAULT_REGULARIZATION_PENALTY);
            session.evaluate(expected, result);
        }
    }
    
    /**
     * Test of call method, of class L1L2.
     */
    @Test
    public void testCallL1_2() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            L1 instance = new L1(tf, 0.02f);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
             float expected = regularizeL1(w, 0.02f);
            session.evaluate(expected, result);
        }
    }
    
    
    
}
