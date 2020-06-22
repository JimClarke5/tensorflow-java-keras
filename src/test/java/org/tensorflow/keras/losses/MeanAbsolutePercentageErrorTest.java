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
public class MeanAbsolutePercentageErrorTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public MeanAbsolutePercentageErrorTest() {
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
    public void testCongig() {
        System.out.println("testCongig");
        MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(null);
        assertEquals("mean_squared_error", instance.getName());

        instance = new MeanAbsolutePercentageError(null, "mape_1", Reduction.SUM);
        assertEquals("mape_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class MeanAbsolutePercentageError.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf);
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_true);
            float expected = 0.0F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class MeanAbsolutePercentageError.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf);
            float[] true_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 211.85184F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class MeanAbsolutePercentageError.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf);
            float[] true_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 487.25922F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 422.8889F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        System.out.println("test_zero_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
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
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf, Reduction.AUTO);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3, 1)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 694.4445F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_no_reduction() {
        System.out.println("test_no_reduction");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf, Reduction.NONE);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            Float[] expected = {621.8518F, 352.66666F};
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sum_reduction() {
        System.out.println("test_sum_reduction");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanAbsolutePercentageError instance = new MeanAbsolutePercentageError(tf, Reduction.SUM);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            System.out.println(loss.asOutput().shape());
            float expected = 974.51843F;
            testSession.evaluate(expected, loss);
        }
    }

}
