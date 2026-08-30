interface FieldProps {
  label: string;
  value: string;
  className?: string;
}

// Extracted once a second tab (Dataset, after Overview) needed the exact
// same dt/dd label-over-value markup — not worth a component until a
// real second use case existed.
export default function Field({ label, value, className = "" }: FieldProps) {
  return (
    <div className={className}>
      <dt className="text-xs font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
        {label}
      </dt>
      <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">{value}</dd>
    </div>
  );
}
