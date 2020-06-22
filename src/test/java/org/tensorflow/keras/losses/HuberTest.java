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
package org.tensorflow.keras.losses;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;

/**
 *
 * @author Jim Clarke
 */
public class HuberTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public HuberTest() {
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

    @Test
    public void testConfig() {
        System.out.println("testConfig");
        Huber instance = new Huber(null);
        assertEquals("huber_loss", instance.getName());

        instance = new Huber(null, "huber", Reduction.SUM);
        assertEquals("huber", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    @Test
    public void test_all_correct() {
        System.out.println("test_all_correct");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Huber instance = new Huber(tf);
            Operand loss = instance.call(y_true, y_true);
            float expected = 0.0F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class Huber.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Huber instance = new Huber(tf);
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.10416666666666669F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class Huber.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Huber instance = new Huber(tf);
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0.23958333333333337F;
            testSession.evaluate(expected, loss);

            // todo Verify we get the same output when the same input is given
        }
    }

    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] sample_narray = {1.2f, 3.4f};
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Huber instance = new Huber(tf);
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0.22766666666666668F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        System.out.println("test_zero_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Huber instance = new Huber(tf);
            Operand sampleWeight = tf.constant(0.F);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_timestep_weighted() {
        System.out.println("test_timestep_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f};
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3, 1)));
            Huber instance = new Huber(tf);
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);

            float expected = .4025F;
            testSession.evaluate(expected, loss);
        }
    }

}
