import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Supplier;

public class Weka {

    public static final int PERCENT_SPLIT = 66;
    public static final long RANDOM_SEED = 1L;

    // Stored best model for prediction
    private ModelSpec bestSpec;
    private Classifier bestModel;

    public interface ProgressCallback {
        void onProgress(int percent, String message);
    }

    enum TransformType {
        NONE,            // Tree-based algorithms: do not change dataset
        DISCRETIZE,      // NaiveBayes: numeric -> nominal
        NUMERIC_PIPELINE // Numeric algorithms: Normalize + NominalToBinary
    }

    static class ModelSpec {
        final String name;
        final Supplier<Classifier> supplier;
        final TransformType transform;
        final String prepLabel;

        ModelSpec(String name, Supplier<Classifier> supplier, TransformType transform, String prepLabel) {
            this.name = name;
            this.supplier = supplier;
            this.transform = transform;
            this.prepLabel = prepLabel;
        }
    }

    public static class EvalRow {
        public final String algo;
        public final String prep;
        public final int correct;
        public final int total;
        public final double accuracy;
        public final String status;

        public EvalRow(String algo, String prep, int correct, int total, double accuracy, String status) {
            this.algo = algo;
            this.prep = prep;
            this.correct = correct;
            this.total = total;
            this.accuracy = accuracy;
            this.status = status;
        }
    }

    public static class RunResult {
        public final List<EvalRow> rows;
        public final EvalRow bestRow;

        public RunResult(List<EvalRow> rows, EvalRow bestRow) {
            this.rows = rows;
            this.bestRow = bestRow;
        }
    }

    public boolean hasBestModel() {
        return bestModel != null;
    }

    // ================= Dataset loading =================
    public Instances loadDataset(File file) throws Exception {
        String n = file.getName().toLowerCase(Locale.ROOT);

        Instances d;
        if (n.endsWith(".csv")) {
            CSVLoader loader = new CSVLoader();
            loader.setSource(file);

            boolean hasHeader = detectCsvHeader(file);
            loader.setNoHeaderRowPresent(!hasHeader);

            d = loader.getDataSet();
        } else {
            DataSource src = new DataSource(file.getAbsolutePath());
            d = src.getDataSet();
        }

        if (d == null) throw new IllegalArgumentException("Dataset could not be read.");
        if (d.numAttributes() < 2) throw new IllegalArgumentException("Dataset must have at least 2 attributes.");

        d.setClassIndex(d.numAttributes() - 1);
        return d;
    }

    private boolean detectCsvHeader(File csv) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(csv), StandardCharsets.UTF_8))) {
            String first = br.readLine();
            if (first == null) return true;

            String[] parts = first.split("[,;\\t]");
            if (parts.length == 0) return true;

            // If any token is NOT a number, assume header exists
            for (String p : parts) {
                String t = p.trim();
                if (t.isEmpty()) return true;
                try { Double.parseDouble(t); }
                catch (Exception ex) { return true; }
            }
            return false;
        } catch (Exception e) {
            return true;
        }
    }

    public void validateDatasetForClassification(Instances d) {
        if (d.classIndex() < 0) d.setClassIndex(d.numAttributes() - 1);

        if (d.classAttribute().isNumeric()) {
            throw new IllegalArgumentException(
                    "Class attribute is NUMERIC (regression). Project expects NOMINAL class for classification."
            );
        }

        // Optional strictness: block unsupported attribute types (string/date/relational)
        for (int i = 0; i < d.numAttributes(); i++) {
            if (i == d.classIndex()) continue;
            if (!(d.attribute(i).isNumeric() || d.attribute(i).isNominal())) {
                throw new IllegalArgumentException(
                        "Unsupported attribute type: " + d.attribute(i).name() +
                                ". Use only NUMERIC or NOMINAL attributes in ARFF/CSV."
                );
            }
        }
    }

    // ================= Model list (>= 10 approaches) =================
    private List<ModelSpec> createSpecs() {
        List<ModelSpec> specs = new ArrayList<>();

        // 5 KNN variants (numeric)
        for (int k : new int[]{1, 3, 5, 7, 9}) {
            specs.add(new ModelSpec(
                    "IBk (k=" + k + ")",
                    () -> new IBk(k),
                    TransformType.NUMERIC_PIPELINE,
                    "Normalize + NominalToBinary"
            ));
        }

        // Naive Bayes: dataset must be nominal -> discretize numeric attributes
        specs.add(new ModelSpec(
                "NaiveBayes",
                NaiveBayes::new,
                TransformType.DISCRETIZE,
                "Discretize"
        ));

        // Logistic: numeric
        specs.add(new ModelSpec(
                "Logistic",
                Logistic::new,
                TransformType.NUMERIC_PIPELINE,
                "Normalize + NominalToBinary"
        ));

        // Tree-based: do NOT change anything
        specs.add(new ModelSpec("J48", J48::new, TransformType.NONE, "Original (no change)"));
        specs.add(new ModelSpec("RandomTree", RandomTree::new, TransformType.NONE, "Original (no change)"));

        specs.add(new ModelSpec(
                "RandomForest (100)",
                () -> {
                    RandomForest rf = new RandomForest();
                    rf.setNumIterations(100);
                    return rf;
                },
                TransformType.NONE,
                "Original (no change)"
        ));

        // ANN + SVM: numeric
        specs.add(new ModelSpec("MultilayerPerceptron", MultilayerPerceptron::new,
                TransformType.NUMERIC_PIPELINE, "Normalize + NominalToBinary"));

        specs.add(new ModelSpec("SMO (SVM)", SMO::new,
                TransformType.NUMERIC_PIPELINE, "Normalize + NominalToBinary"));

        return specs; // total = 12
    }

    // ================= Classifier builder =================
    private Classifier buildClassifierForSpec(ModelSpec spec) throws Exception {
        Classifier base = spec.supplier.get();

        if (spec.transform == TransformType.NONE) {
            return base;
        }

        FilteredClassifier fc = new FilteredClassifier();
        fc.setClassifier(base);

        if (spec.transform == TransformType.DISCRETIZE) {
            fc.setFilter(new Discretize());
            return fc;
        }

        // Numeric pipeline: teacher sample style (Normalize -> NominalToBinary)
        MultiFilter mf = new MultiFilter();
        mf.setFilters(new Filter[]{ new Normalize(), new NominalToBinary() });
        fc.setFilter(mf);

        return fc;
    }

    // ================= Run all models + choose best =================
    public RunResult runAllModels(Instances original, ProgressCallback cb) throws Exception {
        validateDatasetForClassification(original);

        List<ModelSpec> specs = createSpecs();
        List<EvalRow> rows = new ArrayList<>();

        int total = specs.size();
        if (cb != null) cb.onProgress(5, "Starting evaluation...");

        for (int i = 0; i < total; i++) {
            ModelSpec spec = specs.get(i);
            int p = 10 + (int) (80.0 * (i + 1) / total);
            if (cb != null) cb.onProgress(p, "Running " + spec.name + " (" + (i + 1) + "/" + total + ")");

            try {
                rows.add(evaluateOne(spec, original));
            } catch (Exception ex) {
                rows.add(new EvalRow(spec.name, spec.prepLabel, 0, 0, 0.0, "ERROR"));
            }
        }

        // choose best by correct, then accuracy
        EvalRow bestRow = rows.stream()
                .filter(r -> "OK".equals(r.status))
                .max(Comparator.<EvalRow>comparingInt(r -> r.correct)
                        .thenComparingDouble(r -> r.accuracy))
                .orElseThrow(() -> new IllegalStateException("No model evaluated successfully."));

        // train final best model on full dataset (for prediction)
        bestSpec = specs.stream()
                .filter(s -> s.name.equals(bestRow.algo))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("Internal error: best spec not found"));

        if (cb != null) cb.onProgress(95, "Training best model on full dataset...");
        bestModel = buildClassifierForSpec(bestSpec);
        bestModel.buildClassifier(original);

        // sort table rows best-first
        rows.sort((a, b) -> {
            int c = Integer.compare(b.correct, a.correct);
            if (c != 0) return c;
            return Double.compare(b.accuracy, a.accuracy);
        });

        if (cb != null) cb.onProgress(100, "Done ✓");
        return new RunResult(rows, bestRow);
    }

    private EvalRow evaluateOne(ModelSpec spec, Instances original) throws Exception {
        // Copy + randomize (teacher style)
        Instances d = new Instances(original);
        d.setClassIndex(original.classIndex());
        d.randomize(new Random(RANDOM_SEED));

        int trainSize = d.numInstances() * PERCENT_SPLIT / 100;
        int testSize = d.numInstances() - trainSize;

        Instances train = new Instances(d, 0, trainSize);
        Instances test  = new Instances(d, trainSize, testSize);

        Classifier model = buildClassifierForSpec(spec);
        model.buildClassifier(train);

        int correct = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            Instance inst = test.instance(i);
            double pred = model.classifyInstance(inst);
            if (pred == inst.classValue()) correct++;
        }

        double acc = (testSize == 0) ? 0.0 : (100.0 * correct / testSize);
        return new EvalRow(spec.name, spec.prepLabel, correct, testSize, acc, "OK");
    }

    // ================= Prediction helpers =================
    public Instance buildUserInstance(Instances header, Map<Integer, Object> valuesByIndex) {
        DenseInstance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);

        for (int i = 0; i < header.numAttributes(); i++) {
            if (i == header.classIndex()) continue;

            Object v = valuesByIndex.get(i);
            if (v == null) {
                inst.setMissing(i);
                continue;
            }

            if (header.attribute(i).isNumeric()) {
                inst.setValue(i, Double.parseDouble(v.toString()));
            } else {
                inst.setValue(i, v.toString());
            }
        }

        inst.setMissing(header.classIndex());
        return inst;
    }

    public String predictLabel(Instances header, Instance userInst) throws Exception {
        if (bestModel == null) throw new IllegalStateException("Best model is not trained yet.");
        double cls = bestModel.classifyInstance(userInst);
        return header.classAttribute().value((int) cls);
    }
}
