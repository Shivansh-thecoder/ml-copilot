import Field from "@/components/ui/Field";

interface DatasetTabProps {
  datasetName: string;
}

// A project stores its dataset as a plain display name (see
// app/models/project.py's dataset_name), not a link to a real Dataset
// id -- so this tab can't show real rows/columns/target info without
// guessing which uploaded dataset it refers to. Detailed info for a
// specific uploaded dataset is on the Datasets page; the Copilot tab
// works directly with a real dataset_id.
export default function DatasetTab({ datasetName }: DatasetTabProps) {
  return (
    <dl className="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <Field label="Dataset Name" value={datasetName || "—"} />
      <Field
        label="Details"
        value="See the Datasets page for row/column/target information, or use the Copilot tab to select a dataset and run analysis directly."
        className="sm:col-span-2"
      />
    </dl>
  );
}
