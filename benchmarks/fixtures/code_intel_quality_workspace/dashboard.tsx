type DashboardProps = {
  title: string;
};

function useDashboardState() {
  return { ready: true };
}

export function DashboardPanel(props: DashboardProps) {
  const state = useDashboardState();
  return <section>{props.title}:{String(state.ready)}</section>;
}
