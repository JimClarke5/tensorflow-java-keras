/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
public class MeanSquaredErrorTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public MeanSquaredErrorTest() {
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
        MeanSquaredError instance = new MeanSquaredError(null);
        assertEquals("mean_squared_error", instance.getName());

        instance = new MeanSquaredError(null, "mse_1", Reduction.SUM);
        assertEquals("mse_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());
    }

    /**
     * Test of call method, of class MeanSquaredError.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_true);
            float expected = 0.0F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class MeanSquaredError.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            float[] true_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 49.5F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class MeanSquaredError.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            float[] true_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 113.85F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 767.8F / 6F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        System.out.println("test_zero_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
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
            MeanSquaredError instance = new MeanSquaredError(tf, Reduction.AUTO);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3, 1)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);

            float expected = 587F / 6F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_invalid_sample_weight() {
        System.out.println("test_invalid_sample_weight");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] sample_narray = {3f, 6f, 5f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3, 1)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 2)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 83F / 6F;
            testSession.evaluate(expected, loss);
            fail();
        } catch (Exception expected) {

        }
    }

    @Test
    public void test_no_reduction() {
        System.out.println("test_no_reduction");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf, Reduction.NONE);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            Float[] expected = {84.333336F, 143.36665F};
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sum_reduction() {
        System.out.println("test_sum_reduction");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            MeanSquaredError instance = new MeanSquaredError(tf, Reduction.SUM);
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            System.out.println(loss.asOutput().shape());
            Float[] expected = {227.69998F};
            testSession.evaluate(expected, loss);
        }
    }

}
