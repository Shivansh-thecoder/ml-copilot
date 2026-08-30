import SectionHeader from "@/components/ui/SectionHeader";
import Field from "@/components/ui/Field";

// No theme/notification/workspace toggles here -- none of that existed
// as real, working preferences (nothing in the app read them back), and
// showing controls that don't do anything is worse than showing nothing.
// The one thing genuinely worth surfacing is where this frontend is
// actually pointed, since that's real configuration a user might need to
// check (see services/api.ts and .env.local.example).
export default function SettingsPage() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  return (
    <section className="flex flex-1 flex-col overflow-y-auto px-8 py-10">
      <div className="mx-auto w-full max-w-4xl">
        <SectionHeader title="Settings" description="Connection and environment information." />

        <dl className="mt-8 grid max-w-md grid-cols-1 gap-6">
          <Field label="Backend API URL" value={apiUrl} />
        </dl>
      </div>
    </section>
  );
}
