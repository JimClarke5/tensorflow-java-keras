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
package org.tensorflow.keras.utils;

import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public abstract class TestSession implements AutoCloseable {

    protected float epsilon = 1e-5F;
    protected boolean debug;

    public enum Mode {
        EAGER, GRAPH;
    }

    public static TestSession createEagerSession() {
        return new EagerTestSession();
    }

    public static TestSession createGraphSession() {
        return new GraphTestSession();
    }

    public static TestSession createTestSession(Mode mode) {
        return mode == Mode.EAGER ? createEagerSession() : createGraphSession();
    }

    public void initialize() {
        // empty
    }

    public void run(Op op) {

    }

    public <T extends TNumber> void evaluate(Number expected, Operand<T> input) {
        evaluate(new Number[]{expected}, input);
    }

    public <T extends TNumber> void evaluate(Number expected, Op input) {
        evaluate(new Number[]{expected}, input);
    }

    public <T extends TNumber> void evaluate(Number[] expected, Op input) {
        Output output = input.op().output(0);
        evaluate(expected, output);
    }

    public <T extends TNumber> void evaluate(Number[] expected, Operand<T> input) {
        Output output = input.asOutput();
        evaluate(expected, output);
    }

    public abstract <T extends TNumber> void evaluate(Number[] expected, Output<T> input);

    public void evaluate(String expected, Operand<TString> input) {
        evaluate(new String[]{expected}, input);
    }

    public void evaluate(String expected, Op input) {
        evaluate(new String[]{expected}, input);
    }

    public void evaluate(String[] expected, Op input) {
        Output output = input.op().output(0);
        evaluate(expected, output);
    }

    public void evaluate(String[] expected, Operand<TString> input) {
        Output output = input.asOutput();
        evaluate(expected, output);
    }

    public abstract void evaluate(String[] expected, Output<TString> input);

    public void evaluate(Boolean expected, Operand<TBool> input) {
        evaluate(new Boolean[]{expected}, input);
    }

    public void evaluate(Boolean expected, Op input) {
        evaluate(new Boolean[]{expected}, input);
    }

    public void evaluate(Boolean[] expected, Op input) {
        Output output = input.op().output(0);
        evaluate(expected, output);
    }

    public void evaluate(Boolean[] expected, Operand<TBool> input) {
        Output output = input.asOutput();
        evaluate(expected, output);
    }

    public abstract void evaluate(Boolean[] expected, Output<TBool> input);

    public <T extends TType> void evaluate(Operand<T> expected, Op input) {
        Output output = input.op().output(0);
        evaluate(expected, output);
    }

    public <T extends TType> void evaluate(Operand<T> expected, Operand<T> input) {
        evaluate(expected.asOutput(), input.asOutput());
    }

    public abstract <T extends TType> void evaluate(Output<T> expected, Output<T> input);

    public <T extends TType> void print(OutputStream out, Operand<T> input) {
        print(new PrintWriter(new OutputStreamWriter(out)), input.asOutput());
    }

    public <T extends TType> void print(OutputStream out, Op input) {
        print(new PrintWriter(new OutputStreamWriter(out)), input.op().output(0));
    }

    public <T extends TType> void print(OutputStream out, Output<T> input) {
        print(new PrintWriter(new OutputStreamWriter(out)), input);
    }

    public <T extends TType> void print(Writer writer, Operand<T> input) {
        print(new PrintWriter(writer), input.asOutput());
    }

    public <T extends TType> void print(Writer writer, Op input) {
        print(new PrintWriter(writer), input.op().output(0));
    }

    public <T extends TType> void print(Writer writer, Output<T> input) {
        print(new PrintWriter(writer), input);
    }

    public abstract <T extends TType> void print(PrintWriter writer, Output<T> input);

    public abstract Ops getTF();

    public abstract boolean isEager();

    public boolean isGraph() {
        return !isEager();
    }

    public float getEpsilon() {
        return this.epsilon;
    }

    public void setEpsilon(float epsilon) {
        this.epsilon = epsilon;
    }

    public abstract Session getGraphSession();

    public abstract EagerSession getEagerSession();

    @Override
    public abstract void close();

    /**
     * @return the debug
     */
    public boolean isDebug() {
        return debug;
    }

    /**
     * @param debug the debug to set
     */
    public void setDebug(boolean debug) {
        this.debug = debug;
    }

}
