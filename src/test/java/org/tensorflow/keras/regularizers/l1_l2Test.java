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
public class l1_l2Test  extends CommonTest {
    
    public l1_l2Test() {
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
            l1_l2 instance = new l1_l2(tf, 0.2f, 0.3f);
            assertEquals(0.2f, instance.getL1());
            assertEquals(0.3f, instance.getL2());
            
            
            instance = new l1_l2(tf, 0.5f, 0f);
            assertEquals(0.5f, instance.getL1());
            assertEquals(0f, instance.getL2());
            
            instance = new l1_l2(tf, 0f, 0.5f);
            assertEquals(0.f, instance.getL1());
            assertEquals(0.5f, instance.getL2());
            
            instance = new l1_l2(tf);
            assertEquals(Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL1());
            assertEquals(Regularizer.DEFAULT_REGULARIZATION_PENALTY, instance.getL2());
        }
    }

    /**
     * Test of call method, of class l1_l2.
     */
    @Test
    public void testCallZero() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = new l1_l2(tf, 0f, 0f);
            Operand result = instance.call(tf.constant(555));
            session.evaluate(0, result);
        }
    }
    
    
    /**
     * Test of call method, of class l1_l2.
     */
    @Test
    public void testCallNO() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = new l1_l2(tf);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
            float expected = regularizeL1L2(w,
                    Regularizer.DEFAULT_REGULARIZATION_PENALTY, 
                    Regularizer.DEFAULT_REGULARIZATION_PENALTY);
            session.evaluate(expected, result);
        }
    }
    
    
    /**
     * Test of call method, of class l1_l2.
     */
    @Test
    public void testCallL1L2() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = new l1_l2(tf, 0.01f, 0.02f);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
            float expected = regularizeL1L2(w, 0.01f, 0.02f);
            session.evaluate(expected, result);
        }
    }
    
    /**
     * Test of call method, of class l1_l2.
     */
    @Test
    public void testCallL1() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = new l1_l2(tf, 0.01f, 0f);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
            float expected = regularizeL1(w, 0.01f);
            session.evaluate(expected, result);
        }
    }
    
    /**
     * Test of call method, of class l1_l2.
     */
    @Test
    public void testCallL2() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            l1_l2 instance = new l1_l2(tf,0f, 0.02f);
            float [][] w =  {{ 1.0f, 0.9f, 0.8f}, {1.2f, 0.7f, 1.1f}};
            Operand weights = tf.constant(w);
            Operand result = instance.call(weights);
            float expected = regularizeL2(w, 0.02f);
            session.evaluate(expected, result);
        }
    }
    
    
}
